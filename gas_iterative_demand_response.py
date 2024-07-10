"""
cancel out the basic system load and replace it with a convex combination of bids

note: the demand_module (or some subsidiary module) may store calibration data
at the module level (not in the model), so this module should only be used with one
model at a time. An alternative approach would be to receive a calibration_data
object back from demand_module.calibrate(), then add that to the model and pass
it back to the bid function when needed.

note: we also take advantage of this assumption and store a reference to the
current demand_module in this module (rather than storing it in the model itself)
"""

from __future__ import print_function
from __future__ import division

# TODO: This module handles total-cost pricing. 
# Apply a simple tax to every retail MMBtu of gas sold (gas_demand_ref_quantity or FlexibleDemand)
# (this is a fixed adder to the cost in $/MMBtu, not a multiplier times the marginal cost)
# The module can be used to find the right tax to come out
# revenue-neutral (i.e., recover any stranded costs, rebate any supply-side rents)

import os, sys, time
from pprint import pprint
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

try:
    from pyomo.repn import generate_standard_repn
except ImportError:
    # this was called generate_canonical_repn before Pyomo 5.6
    from pyomo.repn import generate_canonical_repn as generate_standard_repn

import switch_model.utilities as utilities

import util

demand_module = None  # will be set via command-line options

def define_arguments(argparser):
    argparser.add_argument(
        "--dr-flat-pricing",
        action="store_true",
        default=False,
        help="Charge a constant (average) price for electricity, rather than varying hour by hour",
    )
    argparser.add_argument(
        "--dr-demand-module",
        default=None,
        help="Name of module to use for demand-response bids. This should also be "
        "specified in the modules list, and should provide calibrate() and bid() functions. "
        "Pre-written options include constant_elasticity_demand_system or r_demand_system. "
        "Specify one of these in the modules list and use --help again to see module-specific options.",
    )

def define_components(m):
    # load scipy.optimize; this is done here to avoid loading it during unit tests
    try:
        global scipy
        import scipy.optimize
    except ImportError:
        print("=" * 80)
        print(
            "Unable to load scipy package, which is used by the demand response system."
        )
        print("Please install this via 'conda install scipy' or 'pip install scipy'.")
        print("=" * 80)
        raise

    ###################
    # Choose the right demand module.
    # NOTE: we assume only one model will be run at a time, so it's safe to store
    # the setting in this module instead of in the model.
    ##################

    global demand_module
    if m.options.dr_demand_module is None:
        raise RuntimeError(
            "No demand module was specified for the demand_response system; unable to continue. "
            "Please use --dr-demand-module <module_name> in options.txt, scenarios.txt or on "
            "the command line. "
            "You should also add this module to the list of modules to load "
            " via modules.txt or --include-module <module_name>."
        )
    if m.options.dr_demand_module not in sys.modules:
        raise RuntimeError(
            "Demand module {mod} cannot be used because it has not been loaded. "
            "Please add this module to the modules list (usually modules.txt) "
            "or specify --include-module {mod} in options.txt, scenarios.txt or "
            "on the command line.".format(mod=m.options.dr_demand_module)
        )
    demand_module = sys.modules[m.options.dr_demand_module]

    # Make sure the model has dual and rc suffixes
    if not hasattr(m, "dual"):
        m.dual = Suffix(direction=Suffix.IMPORT)
    if not hasattr(m, "rc"):
        m.rc = Suffix(direction=Suffix.IMPORT)

    ###################
    # Price Responsive Demand bids
    ##################

    # list of all bids that have been received from the demand system
    m.DR_BID_LIST = Set(dimen=1, initialize=[], ordered=True)
   
    # data for the individual bids; each load_zone gets one bid for each timeseries,
    # So we just record
    # the bid for each timeseries for each gas_zone.
    m.dr_bid = Param(
        m.DR_BID_LIST,
        m.GAS_ZONES,
        m.TIMESERIES,
        mutable=True,
        within=NonNegativeReals,
    )

    # price used to get this bid (only kept for reference)
    m.dr_price = Param(
        m.DR_BID_LIST,
        m.GAS_ZONES,
        m.TIMESERIES,
        mutable=True,
        within=NonNegativeReals,
    )

    # the private benefit of serving each bid
    m.dr_bid_benefit = Param(
        m.DR_BID_LIST, m.GAS_ZONES, m.TIMESERIES, mutable=True, within=Reals
    )

    # weights to assign to the bids for each timeseries when constructing an optimal demand profile
    m.DRBidWeight = Var(
        m.DR_BID_LIST, m.GAS_ZONES, m.TIMESERIES, within=NonNegativeReals
    )

    # choose a convex combination of bids for each zone and timeseries
    m.DR_Convex_Bid_Weight = Constraint(
        m.GAS_ZONES,
        m.TIMESERIES,
        rule=lambda m, z, ts: Constraint.Skip
        if len(m.DR_BID_LIST) == 0
        else (sum(m.DRBidWeight[b, z, ts] for b in m.DR_BID_LIST) == 1),
    )

    # For flat-price models, we have to use the same weight for all timeseries within the
    # same year (period), because there is only one price for the whole period, so it can't
    # induce different adjustments in individual timeseries.
    if m.options.dr_flat_pricing:
        m.DR_Flat_Bid_Weight = Constraint(
            m.DR_BID_LIST,
            m.GAS_ZONES,
            m.TIMESERIES,
            rule=lambda m, b, z, ts: m.DRBidWeight[b, z, ts]
            == m.DRBidWeight[b, z, m.tp_ts[m.TPS_IN_PERIOD[m.ts_period[ts]].first()]],
        )

    # Optimal level of demand, calculated from available bids (negative, indicating consumption)
    m.FlexibleDemand = Expression(
        m.GAS_ZONES,
        m.TIMESERIES,
        rule=lambda m, z, ts: sum(
            m.DRBidWeight[b, z, ts] * m.dr_bid[b, z, ts] for b in m.DR_BID_LIST
        ),
    )

    # replace gas_demand_ref_quantity with FlexibleDemand in the GAS balance constraint
    idx = m.Zone_Gas_Withdrawals.index("gas_demand_ref_quantity")
    m.Zone_Gas_Withdrawals[idx] = "FlexibleDemand"

    # private benefit of the gas consumption
    # (i.e., willingness to pay for the current gas supply)
    # reported as negative cost, i.e., positive benefit
    # also divide by duration of the timeseries
    # to convert from a cost per timeseries to a cost per hour, which can
    # be reported per timepoint.
    m.DR_Welfare_Cost = Expression(
        m.TIMEPOINTS,
        rule=lambda m, tp: (-1.0)
        * sum(
            m.DRBidWeight[b, z, m.tp_ts[tp]] * m.dr_bid_benefit[b, z, m.tp_ts[tp]]
            for b in m.DR_BID_LIST
            for z in m.GAS_ZONES
        )
        / m.ts_duration_hrs[m.tp_ts[tp]],
    )

    # add the private benefit to the model's objective function
    m.Cost_Components_Per_TP.append("DR_Welfare_Cost")

    # variable to store the baseline data
    m.base_data = None

def pre_iterate(m):
   
    # NOTE:
    # bids must be added to the model here, and the model must be reconstructed here,
    # so the model can then be solved and remain in a "solved" state through the end
    # of post-iterate, to avoid problems in final reporting.

    # store various properties from previous model solution for later reference
    if m.iteration_number == 0:
        # model hasn't been solved yet
        m.prev_marginal_cost = {
            (z, ts): None for z in m.GAS_ZONES for ts in m.TIMESERIES
        }
        m.prev_demand = {(z, ts): None for z in m.GAS_ZONES for ts in m.TIMESERIES}
        m.prev_SystemCost = None
    else:
        # get values from previous solution
        m.prev_marginal_cost = {
            (z, ts): gas_marginal_cost(m, z, ts)
            for z in m.GAS_ZONES
            for ts in m.TIMESERIES
        }
        m.prev_demand = {
            (z, ts): gas_demand(m, z, ts) for z in m.GAS_ZONES for ts in m.TIMESERIES
        }
        m.prev_SystemCost = value(m.SystemCost)

    if m.iteration_number > 0:
        # store cost of previous solution before it gets altered by update_demand()
        # TODO: this and best_cost could probably be moved to post_iterate
        # Then we'd be comparing the final (current) solution to the best possible
        # solution based on the prior round of bids, rather than comparing the new
        # bid to the prior solution to the master problem. This is probably fine.
        # TODO: does this correctly account for producer surplus? It seems like that's
        # being treated as a cost (embedded in MC * demand); maybe this should use
        # total direct cost instead,
        # or focus specifically on consumer surplus (use prices instead of MC as the
        # convergence measure). But maybe this is OK, since the question is, "if we
        # could serve the last bid at the MC we had then (which also means the PS
        # we had then? no change for altered volume?), would everyone be much
        # better off than they are with the allocation we have now chosen?"
        # Maybe using MC lets us focus on whether there can be another incrementally
        # different solution that would be much better than the one we have now.
        # This ignores other solutions far away, where an integer variable is flipped,
        # but that's OK. (?)
        prev_direct_cost = value(
            sum(
                sum(
                    m.prev_marginal_cost[z, ts] * m.prev_demand[z, ts]
                    for z in m.GAS_ZONES
                )
                * m.ts_scale_to_year[ts]
                * m.bring_annual_costs_to_base_year[m.ts_period[ts]]
                for ts in m.TIMESERIES
            )
        )
        prev_welfare_cost = value(
            sum(
                m.DR_Welfare_Cost[tp] * m.bring_timepoint_costs_to_base_year[tp]
                for tp in m.TIMEPOINTS
            )
        )
        prev_cost = prev_direct_cost + prev_welfare_cost

        print("")
        print("previous direct cost: ${:,.0f}".format(prev_direct_cost))
        print("previous welfare cost: ${:,.0f}".format(prev_welfare_cost))
        print("")

    # get the next bid and attach it to the model
    update_demand(m)

    if m.iteration_number > 0:
        # get an estimate of best possible net cost of serving gas
        # (if we could completely serve the last bid at the prices we quoted,
        # that would be an optimum; the actual cost may be higher but never lower)
        b = m.DR_BID_LIST.last()  # current bid number
        best_direct_cost = value(
            sum(
                sum(
                    m.prev_marginal_cost[z, ts] * m.dr_bid[b, z, ts]
                    for z in m.GAS_ZONES
                )
                * m.ts_scale_to_year[ts]
                * m.bring_annual_costs_to_base_year[m.ts_period[ts]]
                for ts in m.TIMESERIES
            )
        )

        best_bid_benefit = value(
            sum(
                -sum(m.dr_bid_benefit[b, z, ts] for z in m.GAS_ZONES)
                * m.ts_scale_to_year[ts]
                * m.bring_annual_costs_to_base_year[m.ts_period[ts]]
                for ts in m.TIMESERIES
            )
        )
        best_cost = best_direct_cost + best_bid_benefit

        print("")
        print("best direct cost: ${:,.0f}".format(best_direct_cost))
        print("best bid benefit: ${:,.0f}".format(best_bid_benefit))
        print("")

        print(
            "lower bound=${:,.0f}, previous cost=${:,.0f}, optimality gap (vs direct cost)={}".format(
                best_cost, prev_cost, (prev_cost - best_cost) / abs(prev_direct_cost)
            )
        )
        if prev_cost < best_cost:
            print(
                "WARNING: final cost is below reported lower bound; "
                "there is probably a problem with the demand system."
            )

        # import pdb; pdb.set_trace()

    # basis for optimality test:

    # 1. The total cost of supply, as a function of quantity produced each hour, forms
    # a surface which is convex downward, since it is linear (assuming all variables are
    # continuous or all integer variables are kept at their current level, i.e., the curve
    # is locally convex). (Think of the convex hull of the extreme points of the production
    # cost function.)

    # 2. The total benefit of consumption, as a function of quantity consumed each hour,
    # forms a surface which is convex upward (by the assumption/requirement of convexity
    # of the demand function).

    # 3. marginal costs (prev_marginal_cost) and production levels (prev_demand) from the
    # most recent solution to the master problem define a production cost plane which is
    # tangent to the production cost function at that production level. From 1, the production
    # cost function must lie on or above this surface everywhere. This plane is given by

    # (prev_SystemCost - prev_welfare_cost) + prev_marginal_cost * (demand - prev_demand)

    # 4. The last bid quantities (dr_bid[-1]) must be at a point where marginal benefit of consumption
    # equals marginal cost of consumption (prev_marginal_cost) in all directions; otherwise
    # they would not be a private optimum.

    # 5. The benefit reported in the last bid (dr_bid_benefit[-1]) shows the level of the total
    # benefit curve at that point.

    # 6. From 2, 4 and 5, the prev_marginal_cost and the last reported benefit must form
    # a plane which is at or above the total benefit curve everywhere. This plane is given by

    # dr_bid_benefit[-1] + (demand - dr_bid[-1]) * dr_price[-1]
    # dr_bid_benefit[-1] + (demand - dr_bid[-1]) * prev_marginal_cost

    # 7. Since the total cost curve must lie above the plane defined in 3. and the total
    # benefit curve must lie below the plane defined in 6., the (constant) distance between
    # these planes is an upper bound on the net benefit that can be obtained, or with a change
    # of sign, a lower bound on the (negative) "cost" that can be achieved. In other words,

    # best_SystemCost >=
    # (prev_SystemCost - prev_welfare_cost) + prev_marginal_cost * (demand - prev_demand)
    # - (dr_bid_benefit[-1] + (demand - dr_bid[-1]) * prev_marginal_cost)
    # so, best_SystemCost >= prev_SystemCost - prev_welfare_cost - prev_marginal_cost * prev_demand - dr_bid_benefit[-1] + dr_bid[-1] * prev_marginal_cost
    # so, best_SystemCost >= prev_SystemCost - prev_welfare_cost - dr_bid_benefit[-1] + prev_marginal_cost * (dr_bid[-1] - prev_demand)

    # so, prev_SystemCost - best_SystemCost <= (prev_marginal_cost * prev_demand + prev_welfare_cost) - (prev_marginal_cost * dr_bid[-1] - dr_bid_benefit[-1])

    # from definitions above:
    # prev_cost = prev_direct_cost + prev_welfare_cost = prev_marginal_cost * prev_demand + prev_welfare_cost
    # best_cost = best_direct_cost + best_bid_benefit = prev_marginal_cost * m.dr_bid[-1] - dr_bid_benefit[-1]

    # so, the best possible improvement (optimality gap) is
    # prev_SystemCost - best_SystemCost <= prev_cost - best_cost

    # where prev_cost is the best cost the system operator could come up with
    # for customers so far (publicly optimal consumption at these prices based
    # on prior bids) and best_cost is the request the customers made in response
    # (privately optimal consumption at those prices)

    # Check for convergence -- optimality gap is less than 0.01% of most
    # recently offered total cost (which may be negative?)
    converged = (
        m.iteration_number > 0
        and abs(prev_cost - best_cost) / abs(prev_direct_cost) <= 0.0001
    )

    return converged

def post_iterate(m):
    print("\n\n=======================================================")
    print("Solved model")
    print("=======================================================")
    print("Total cost: ${v:,.0f}".format(v=value(m.SystemCost)))

    # TODO:
    # maybe calculate prices for the next round here and attach them to the
    # model, so they can be reported as final prices (currently we don't
    # report the final prices, only the prices prior to the final model run)

    SystemCost = value(m.SystemCost)  # calculate once to save time
    if m.prev_SystemCost is None:
        print(
            "prev_SystemCost=<n/a>, SystemCost={:,.0f}, ratio=<n/a>".format(SystemCost)
        )
    else:
        print(
            "prev_SystemCost={:,.0f}, SystemCost={:,.0f}, ratio={}".format(
                m.prev_SystemCost, SystemCost, SystemCost / m.prev_SystemCost
            )
        )

    tag = filename_tag(m, include_iter_num=False)
    outputs_dir = m.options.outputs_dir

    # report information on most recent bid
    if m.iteration_number == 0:
        util.create_table(
            output_file=os.path.join(outputs_dir, f"bid{tag}.csv"),
            headings=(
                "bid_num",
                "gas_zone",
                "timeseries",
                "marginal_cost",
                "price",
                "bid",
                "wtp",
                "base_price",
                "base_load",
            ),
        )
    b = m.DR_BID_LIST.last()  # current bid
    util.append_table(
        m,
        m.GAS_ZONES,
        m.TIMESERIES,
        output_file=os.path.join(outputs_dir, f"bid{tag}.csv"),
        values=lambda m, z, ts: (
            b,
            z,
            ts,
            m.prev_marginal_cost[z, ts],
            m.dr_price[b, z, ts],
            m.dr_bid[b, z, ts],
            m.dr_bid_benefit[b, z, ts],
            m.base_data_dict[z, ts][1],
            m.base_data_dict[z, ts][0],
        ),
    )

    # store the current bid weights for future reference
    if m.iteration_number == 0:
        util.create_table(
            output_file=os.path.join(outputs_dir, f"bid_weights{tag}.csv"),
            headings=("iteration", "gas_zone", "timeseries", "bid_num", "weight"),
        )
    util.append_table(
        m,
        m.GAS_ZONES,
        m.TIMESERIES,
        m.DR_BID_LIST,
        output_file=os.path.join(outputs_dir, f"bid_weights{tag}.csv"),
        values=lambda m, z, ts, b: (
            len(m.DR_BID_LIST),
            z,
            ts,
            b,
            m.DRBidWeight[b, z, ts],
        ),
    )

    # if m.iteration_number % 5 == 0:
    #     # save time by only writing results every 5 iterations
    # write_results(m)

    # Stop if there are no duals. This is an efficient point to check, and
    # otherwise the errors later are pretty cryptic.
    if not m.dual:
        raise RuntimeError(
            "No dual values have been calculated. Check that your solver is "
            "able to provide duals for integer programs. If using cplex, you "
            "may need to specify --retrieve-cplex-mip-duals."
        )

    # write_dual_costs(m)
    write_results(m)
    write_batch_results(m)

    # if m.iteration_number >= 3:
    #     import pdb; pdb.set_trace()


def update_demand(m):
    """
    This should be called after solving the model, in order to calculate new bids
    to include in future runs. The first time through, it also uses the fixed demand
    and marginal costs to calibrate the demand system, and then replaces the fixed
    demand with the flexible demand system.
    """
    first_run = m.base_data is None

    print("attaching new demand bid to model")
    if first_run:
        calibrate_model(m)
    else:  # not first run
        if m.options.verbose and len(m.GAS_ZONES) * len(m.TIMESERIES) <= 20:
            print("m.DRBidWeight:")
            pprint(
                [
                    (
                        z,
                        ts,
                        [(b, value(m.DRBidWeight[b, z, ts])) for b in m.DR_BID_LIST],
                    )
                    for z in m.GAS_ZONES
                    for ts in m.TIMESERIES
                ]
            )

    # get new bids from the demand system at the current prices
    bids = get_bids(m)

    # add the new bids to the model
    if m.options.verbose:
        print("adding bids to model")
        # print("first day (z, ts, prices, demand, wtp) =")
        # pprint(bids[0])
    add_bids(m, bids)

    log_infeasible_constraints(m)

def total_direct_costs_per_year(m, period):
    """Return undiscounted total cost per year, during each period, as calculated by Switch,
    including everything except DR_Welfare_Cost.

    This code comes from financials.calc_sys_costs_per_period(), excluding discounting
    and upscaling to the period.

    NOTE: ideally this would give costs by zone and period, to allow calibration for different
    utilities within a large study. But the cost components don't distinguish that way.
    (They probably should, since that would allow us to discuss average electricity costs
    in each zone.)
    """
    return value(
        sum(
            getattr(m, annual_cost)[period]
            for annual_cost in m.Cost_Components_Per_Period
        )
        + sum(
            getattr(m, tp_cost)[t] * m.tp_weight_in_year[t]
            for t in m.TPS_IN_PERIOD[period]
            for tp_cost in m.Cost_Components_Per_TP
            if tp_cost != "DR_Welfare_Cost"
        )
    )

def gas_marginal_cost(m, z, ts):
    """Return marginal cost of providing product prod in gas_zone z during timeseries ts."""
    component = m.Zone_Gas_Balance[z, ts]
    return m.dual[component] / (
        m.bring_annual_costs_to_base_year[m.ts_period[ts]] * m.ts_scale_to_year[ts]
    )


def gas_demand(m, z, ts):
    """Return total consumption of gas in gas_zone z during timeseries ts."""
    if len(m.DR_BID_LIST) == 0:
        # use gas_demand_ref_quantity (base demand) if no bids have been received yet
        # (needed to find flat prices before solving the model the first time)
        demand = m.gas_demand_ref_quantity[z, ts]
    else:
        demand = m.FlexibleDemand[z, ts]
    return demand


def calibrate_model(m):
    """
    Calibrate the demand system and add it to the model.
    """
    # base_data consists of a list of tuples showing (gas_zone, timeseries, base_load (list) and base_price)
    # note: the constructor below assumes list comprehensions will preserve the order of the underlying list
    # (which is guaranteed according to http://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed)
    
    # TODO: add in something for the fixed costs, to make marginal cost commensurate with the base_price
    m.base_data = [
        (
            z,
            ts,
            [m.gas_demand_ref_quantity[z, ts]],
            [m.gas_ref_price[z, ts]],
        )
        for z in m.GAS_ZONES
        for ts in m.TIMESERIES
    ]

    # make a dict of base_data, indexed by load_zone and timepoint, for later reference
    m.base_data_dict = {
        (z, ts): (m.gas_demand_ref_quantity[z, ts], m.gas_ref_price[z, ts])
        for z in m.GAS_ZONES
        for ts in m.TIMESERIES
    }

    # calibrate the demand module
    demand_module.calibrate(m, m.base_data)


def get_prices(m, flat_revenue_neutral=True):
    """Calculate appropriate prices for each day, based on the current state
    of the model."""

    # construct dictionaries of marginal cost vectors for gas for each load zone and time series
    if m.iteration_number == 0:
        # use base prices on the first pass ($0 for everything other than energy)
        marginal_costs = {
            (z, ts): [m.base_data_dict[z, ts][1]]  # one price per timeseries
            for z in m.GAS_ZONES
            for ts in m.TIMESERIES
        }
    else:
        # use marginal costs from last solution
        marginal_costs = {
            (z, ts): [gas_marginal_cost(m, z, ts)]  # one price per timeseries
            for z in m.GAS_ZONES
            for ts in m.TIMESERIES
        }

    if m.options.dr_flat_pricing:
        # find flat price for the whole period that is revenue neutral with the marginal costs
        # (e.g., an aggregator could buy at dynamic marginal cost and sell to the customer at
        # a flat price; the aggregator must find the correct flat price so they break even)
        prices = find_flat_prices(m, marginal_costs, flat_revenue_neutral)
    else:
        prices = marginal_costs
    return prices


def get_bids(m):
    """Get bids from the demand system showing quantities at the current prices and willingness-to-pay for those quantities
    call bid() with dictionary of prices for different products

    Each bid is a tuple of (load_zone, timeseries, {prod: [hourly prices]}, {prod: [hourly quantities]}, wtp)
    quantity will be positive for consumption, negative if customer will supply product
    """

    prices = get_prices(m)

    # get bids for all load zones and timeseries
    bids = []
    for z in m.GAS_ZONES:
        for ts in m.TIMESERIES:
            demand, wtp = demand_module.bid(m, z, ts, prices[z, ts])
            bids.append((z, ts, prices[z, ts], demand, wtp))
    return bids


def find_flat_prices(m, marginal_costs, revenue_neutral):
    # calculate flat prices for an imaginary load-serving entity (LSE) who
    # must break even in each load zone and period.
    # LSE buys at marginal cost, sells at flat prices
    # this is like a transformation on the demand function, where we are
    # now  selling to the LSE rather than directly to the customers
    #
    # LSE iterates in sub-loop (scipy.optimize.newton) to find flat price:
    # set price (e.g., simple average of MC or avg weighted by expected demand)
    # offer price to demand side
    # receive bids
    # calc revenue balance for LSE (q*price - q.MC)
    # if > 0: decrease price (q will go up across the board)
    # if < 0: increase price (q will go down across the board) but

    flat_prices = dict()
    for z in m.GAS_ZONES:
        for p in m.PERIODS:
            price_guess = value(
                sum(
                    marginal_costs[z, ts][0]  # get the only price for this timeseries
                    * gas_demand(m, z, ts)
                    * m.ts_scale_to_year[ts]
                    for ts in m.TS_IN_PERIOD[p]
                )
                / sum(
                    gas_demand(m, z, ts) * m.ts_scale_to_year[ts]
                    for ts in m.TS_IN_PERIOD[p]
                )
            )

            if revenue_neutral:
                # find a flat price that produces revenue equal to marginal costs
                flat_prices[z, p] = scipy.optimize.newton(
                    revenue_imbalance, price_guess, args=(m, z, p, marginal_costs)
                )
            else:
                # used in final round, when LSE is considered to have
                # bought the final constructed quantity at the final
                # marginal cost
                flat_prices[z, p] = price_guess

    # construct a collection of flat prices with the right structure
    final_prices = {
        (z, ts): [flat_prices[z, p]]  # one price per timeseries
        for z in m.GAS_ZONES
        for p in m.PERIODS
        for ts in m.TS_IN_PERIOD[p]
    }
    return final_prices


def revenue_imbalance(flat_price, m, gas_zone, period, dynamic_prices):
    """find demand and revenue that would occur in this gas_zone and period with flat prices, and
    compare to the cost of meeting that demand by purchasing power at the current dynamic prices
    """
    flat_price_revenue = 0.0
    dynamic_price_revenue = 0.0
    for ts in m.TS_IN_PERIOD[period]:
        prices = [flat_price]  # one price per timeseries
        demand, wtp = demand_module.bid(m, gas_zone, ts, prices)
        flat_price_revenue += flat_price * sum(
            d * m.ts_scale_to_year[ts] for d in demand
        )
        dynamic_price_revenue += sum(
            p * d * m.ts_scale_to_year[ts]
            for p, d in zip(dynamic_prices[gas_zone, ts], demand)
        )
    imbalance = dynamic_price_revenue - flat_price_revenue

    print(
        "{}, {}: price ${} produces revenue imbalance of ${}/year".format(
            gas_zone, period, flat_price, imbalance
        )
    )

    return imbalance


def reconstruct(component):
    # reconstruct component, following advice from pyomo/core/base/component.py:538 in Pyomo 6.4.2
    # (.reconstruct method was removed in Pyomo 6.0)
    component.clear()
    component._constructed = False
    component.construct()


def add_bids(m, bids):
    """
    accept a list of bids written as tuples like
    (z, ts, prod, prices, demand, wtp)
    where z is the load zone, ts is the timeseries, prod is the product,
    demand is a list of demand levels for the timepoints during that series (possibly negative, to sell),
    and wtp is the net private benefit from consuming/selling the amount of power in that bid.
    Then add that set of bids to the model
    """

    # create a bid ID
    if len(m.DR_BID_LIST) == 0:
        b = 1
    else:
        b = max(m.DR_BID_LIST) + 1

    # Check that new bids don't violate convexity requirement (there should be
    # no prior bid that appears better at the same prices)
    non_convex_pairs = []
    for z, ts, prices, demand, wtp in bids:
        for prior_b in m.DR_BID_LIST:
            prior_wtp = value(m.dr_bid_benefit[prior_b, z, ts])
            prior_demand = value(m.dr_bid[prior_b, z, ts])
            prior_price = value(m.dr_price[prior_b, z, ts])
            # check for non-convexity, with a little slack
            if (
                prior_wtp - prior_demand * prices[0]
                > wtp - demand[0] * prices[0] + 0.000001
            ):
                # prior bid b looks more attractive than the current bid
                non_convex_pairs.append(
                    f"zone {z}, timeseries {ts}: "
                    f"bid #{prior_b} (made for price {prior_price}) gives more "
                    f"net benefit than bid #{b} at price #{b} ({prices[0]}):\n"
                    f"    {prior_wtp} - {prior_demand} * {prices[0]} > {wtp} - {demand[0]} * {prices[0]}"
                )
        if non_convex_pairs:
            raise (
                ValueError(
                    f'Non-convex bid{"s" if len(non_convex_pairs > 1) else ""} received:\n'
                    + "\n".join(non_convex_pairs)
                    + "\n\nThese indicate non-convexity in the demand bidding function that "
                    + "will prevent the model from converging."
                )
            )

    # extend the list of bids
    m.DR_BID_LIST.add(b)

    # add the bids for each load zone and timepoint to the dr_bid list
    for z, ts, prices, demand, wtp in bids:
        # record the private benefit
        m.dr_bid_benefit[b, z, ts] = wtp
        # record the level of demand for each timepoint
        m.dr_bid[b, z, ts] = demand[0]  # use the single value for this timeseries
        m.dr_price[b, z, ts] = prices[0]  # use the single value for this timeseries

    print("len(m.DR_BID_LIST): {l}".format(l=len(m.DR_BID_LIST)))
    print("m.DR_BID_LIST: {b}".format(b=[x for x in m.DR_BID_LIST]))

    # reconstruct the components that depend on m.DR_BID_LIST, m.dr_bid_benefit and m.dr_bid
    reconstruct(m.DRBidWeight)
    reconstruct(m.DR_Convex_Bid_Weight)
    # reconstruct(m.DR_Gas_Zone_Shared_Bid_Weight)  # obsolete?
    if hasattr(m, "DR_Flat_Bid_Weight"):
        reconstruct(m.DR_Flat_Bid_Weight)
    reconstruct(m.FlexibleDemand)
    
    reconstruct(m.DR_Welfare_Cost)
    # it seems like we have to reconstruct the higher-level components that depend on these
    # ones (even though these are Expressions), because otherwise they refer to objects that
    # used to be returned by the Expression but aren't any more (e.g., versions of DRBidWeight
    # that no longer exist in the model).
    # (i.e., Gas_Balance refers to the items returned by FlexibleDemand instead of referring
    # to FlexibleDemand itself)
    reconstruct(m.Zone_Gas_Balance)
    if hasattr(m, "Aggregate_Spinning_Reserve_Details"):
        reconstruct(m.Aggregate_Spinning_Reserve_Details)
    if hasattr(m, "Satisfy_Spinning_Reserve_Up_Requirement"):
        reconstruct(m.Satisfy_Spinning_Reserve_Up_Requirement)
        reconstruct(m.Satisfy_Spinning_Reserve_Down_Requirement)
    reconstruct(m.SystemCostPerPeriod)
    reconstruct(m.SystemCost)


def reconstruct_gas_balance(m):
    """Reconstruct Energy_Balance constraint, preserving dual values (if present)."""
    # copy the existing Energy_Balance object
    old_Gas_Balance = dict(m.Zone_Gas_Balance)
    reconstruct(m.Zone_Gas_Balance)
    # TODO: now that this happens just before a solve, there may be no need to
    # preserve duals across the reconstruct().
    if m.iteration_number > 0:
        for k in old_Gas_Balance:
            # change dual entries to match new Gas_Balance objects
            m.dual[m.Zone_Gas_Balance[k]] = m.dual.pop(old_Gas_Balance[k])

    # log_infeasible_constraints(m)

def write_batch_results(m):
    # append results to the batch results file, creating it if needed
    output_file = os.path.join(m.options.outputs_dir, "demand_response_summary.csv")

    # create a file to hold batch results if it doesn't already exist
    # note: we retain this file across scenarios so it can summarize all results,
    # but this means it needs to be manually cleared before launching a new
    # batch of scenarios (e.g., when running get_scenario_data or clearing the
    # scenario_queue directory)
    if not os.path.isfile(output_file):
        util.create_table(output_file=output_file, headings=summary_headers(m))

    util.append_table(m, output_file=output_file, values=lambda m: summary_values(m))


def summary_headers(m):
    return (
        ("tag", "iteration", "total_cost")
        + tuple("total_direct_costs_per_year_" + str(p) for p in m.PERIODS)
        + tuple("DR_Welfare_Cost_" + str(p) for p in m.PERIODS)
        + tuple("payment " + str(p) for p in m.PERIODS)
        + tuple("sold " + str(p) for p in m.PERIODS)
    )


def summary_values(m):
    demand_components = [
        c for c in ("gas_demand_ref_quantity", "FlexibleDemand") if hasattr(m, c)
    ]
    values = []

    # tag (configuration)
    values.extend(
        [
            m.options.scenario_name,
            m.iteration_number,
            m.SystemCost,  # total cost (all periods)
        ]
    )

    # direct costs (including "other")
    values.extend([total_direct_costs_per_year(m, p) for p in m.PERIODS])

    # DR_Welfare_Cost
    values.extend(
        [
            sum(
                m.DR_Welfare_Cost[tp] * m.ts_scale_to_year[m.tp_ts[tp]]
                for tp in m.TPS_IN_PERIOD[p]
            )
            for p in m.PERIODS
        ]
    )

    # payments by customers ([expected demand] * [price offered for that demand])
    # note: this uses the final MC to set the final price, rather than using the
    # final price offered to customers. This creates consistency between the final
    # quantities and prices. Otherwise, we would use prices that differ from the
    # final cost by some random amount, and the split between PS and CS would
    # jump around randomly.
    # note: if switching to using the offered prices, then you may have to use None
    # as the customer payment during iteration 0, since m.dr_price[last_bid, z, tp, prod]
    # may not be defined yet.
    last_bid = m.DR_BID_LIST.last()
    values.extend(
        [
            sum(
                # we assume customers pay final marginal cost, so we don't artificially
                # electricity_demand(m, z, tp, prod) * m.dr_price[last_bid, z, tp, prod] * m.tp_weight_in_year[tp]
                gas_demand(m, z, ts)
                * gas_marginal_cost(m, z, ts)
                * m.ts_scale_to_year[ts]
                for z in m.GAS_ZONES
                for ts in m.TS_IN_PERIOD[p]
            )
            for p in m.PERIODS
        ]
    )
    # import pdb; pdb.set_trace()

    # total quantities bought (or sold) by customers each year
    values.extend(
        [
            sum(
                gas_demand(m, z, ts) # * m.ts_scale_to_year[ts]
                for z in m.GAS_ZONES
                for ts in m.TS_IN_PERIOD[p]
            )
            for p in m.PERIODS
        ]
    )

    return values


def get(component, idx, default):
    try:
        return component[idx]
    except KeyError:
        return default


def write_results(m, include_iter_num=True):
    outputs_dir = m.options.outputs_dir
    tag = filename_tag(m, include_iter_num)

    avg_ts_scale = float(sum(m.ts_scale_to_year[ts] for ts in m.TIMESERIES)) / len(
        m.TIMESERIES
    )
    last_bid = m.DR_BID_LIST.last()

    # get final prices that will be charged to customers (not necessarily
    # the same as the final prices they were offered, if iteration was
    # stopped before complete convergence)
    final_prices_by_timeseries = get_prices(m, flat_revenue_neutral=False)
    final_prices = {
        (z, ts): final_prices_by_timeseries[z, ts][0]  # single value per timeseries
        for z in m.GAS_ZONES
        for ts in m.TIMESERIES
    }
    final_quantities = {
        (z, ts): value(
            sum(m.DRBidWeight[b, z, ts] * m.dr_bid[b, z, ts] for b in m.DR_BID_LIST)
        )
        for z in m.GAS_ZONES
        for ts in m.TIMESERIES
    }

    util.write_table(
        m,
        m.GAS_ZONES,
        m.TIMESERIES,
        output_file=os.path.join(outputs_dir, "energy_sources{t}.csv".format(t=tag)),
        headings=("gas_zone", "period", "timeseries")
        + tuple(m.Zone_Gas_Injections)
        + tuple(m.Zone_Gas_Withdrawals)
        + ("offered price", "bid q", "final mc", "final price", "final q")
        + ("peak_day", "base_load", "base_price"),
        values=lambda m, z, t: (z, m.ts_period[t], t)
        + tuple(getattr(m, component)[z, t] for component in m.Zone_Gas_Injections)
        + tuple(getattr(m, component)[z, t] for component in m.Zone_Gas_Withdrawals)
        + (
            m.dr_price[last_bid, z, t],
            m.dr_bid[last_bid, z, t],
            gas_marginal_cost(m, z, t),
            final_prices[z, t],
            final_quantities[z, t],
        )
        + (
            "peak" if m.ts_scale_to_year[t] < 0.5 * avg_ts_scale else "typical",
            m.base_data_dict[z, t][0],
            m.base_data_dict[z, t][1],
        ),
    )

    # import pprint
    # b=[(g, pe, value(m.BuildGen[g, pe]), m.gen_tech[g], m.gen_overnight_cost[g, pe]) for (g, pe) in m.BuildGen if value(m.BuildGen[g, pe]) > 0]
    # bt=set(x[3] for x in b) # technologies
    # pprint([(t, sum(x[2] for x in b if x[3]==t), sum(x[4] for x in b if x[3]==t)/sum(1.0 for x in b if x[3]==t)) for t in bt])


def write_dual_costs(m, include_iter_num=True):
    outputs_dir = m.options.outputs_dir
    tag = filename_tag(m, include_iter_num)

    # with open(os.path.join(outputs_dir, "producer_surplus{t}.csv".format(t=tag)), 'w') as f:
    #     for g, per in m.Max_Build_Potential:
    #         const = m.Max_Build_Potential[g, per]
    #         surplus = const.upper() * m.dual[const]
    #         if surplus != 0.0:
    #             f.write(','.join([const.name, str(surplus)]) + '\n')
    #     # import pdb; pdb.set_trace()
    #     for g, year in m.BuildGen:
    #         var = m.BuildGen[g, year]
    #         if var.ub is not None and var.ub > 0.0 and value(var) > 0.0 and var in m.rc and m.rc[var] != 0.0:
    #             surplus = var.ub * m.rc[var]
    #             f.write(','.join([var.name, str(surplus)]) + '\n')

    outfile = os.path.join(outputs_dir, "dual_costs{t}.csv".format(t=tag))
    dual_data = []
    start_time = time.time()
    print("Writing {} ... ".format(outfile), end=" ")

    def add_dual(const, lbound, ubound, duals, prefix="", offset=0.0):
        if const in duals:
            dual = duals[const]
            if dual >= 0.0:
                direction = ">="
                bound = lbound
            else:
                direction = "<="
                bound = ubound
            if bound is None:
                # Variable is unbounded; dual should be 0.0 or possibly a tiny non-zero value.
                if not (-1e-5 < dual < 1e-5):
                    raise ValueError(
                        "{} has no {} bound but has a non-zero dual value {}.".format(
                            const.name, "lower" if dual > 0 else "upper", dual
                        )
                    )
            else:
                total_cost = dual * (bound + offset)
                if total_cost != 0.0:
                    dual_data.append(
                        (
                            prefix + const.name,
                            direction,
                            (bound + offset),
                            dual,
                            total_cost,
                        )
                    )

    for comp in m.component_objects(ctype=Var):
        for idx in comp:
            var = comp[idx]
            if var.value is not None:  # ignore vars that weren't used in the model
                if var.is_integer() or var.is_binary():
                    # integrality constraint sets upper and lower bounds
                    add_dual(var, value(var), value(var), m.rc, prefix="integer: ")
                else:
                    add_dual(var, var.lb, var.ub, m.rc)
    for comp in m.component_objects(ctype=Constraint):
        for idx in comp:
            constr = comp[idx]
            if constr.active:
                offset = 0.0
                # cancel out any constants that were stored in the body instead of the bounds
                # (see https://groups.google.com/d/msg/pyomo-forum/-loinAh0Wx4/IIkxdfqxAQAJ)
                # (might be faster to do this once during model setup instead of every time)
                standard_constraint = generate_standard_repn(constr.body)
                if standard_constraint.constant is not None:
                    offset = -standard_constraint.constant
                add_dual(
                    constr,
                    value(constr.lower),
                    value(constr.upper),
                    m.dual,
                    offset=offset,
                )

    dual_data.sort(key=lambda r: (not r[0].startswith("DR_Convex_"), r[3] >= 0) + r)

    with open(outfile, "w") as f:
        f.write(
            ",".join(["constraint", "direction", "bound", "dual", "total_cost"]) + "\n"
        )
        f.writelines(",".join(map(str, r)) + "\n" for r in dual_data)
    print("time taken: {dur:.2f}s".format(dur=time.time() - start_time))


def filename_tag(m, include_iter_num=True):
    tag = ""
    if m.options.scenario_name:
        tag += "_" + m.options.scenario_name
    if include_iter_num:
        if m.options.max_iter is None:
            n_digits = 4
        else:
            n_digits = len(str(m.options.max_iter))
            # n_digits = len(str(m.options.max_iter - 1))
        tag += "".join(f"_{t:0{n_digits}d}" for t in m.iteration_node)
    return tag

def post_solve(m, outputs_dir):
    # report final results, possibly after smoothing,
    # and without the iteration number
    write_dual_costs(m, include_iter_num=False)
    write_results(m, include_iter_num=False)
