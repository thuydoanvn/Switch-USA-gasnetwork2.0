This version of SWITCH-GAS **endogenizes natural gas production (domestic supply) and demand**. 
In particular, with the 'gas_wells_build' module, the model will decide the optimal number of gas wells 
by type to be drilled and completed in each investment period. Once a gas well is completed, 
it will produce natural gas following a predetermined production curve. 
Demand data is calibrated when running the 'gas_iterative_demand_response' module with 
price elasticity indicated in 'gas_constant_elasticity_demand_system'.

These data inputs included here are updated to the year 2022 and intended to use in optimization for the period 2023-2025.
Note that base daily demand quantity and price are estimated based on 
reference case projected by the U.S. EIA's Annual Energy Outlook 2023.
Users may want to update data inputs to solve for other periods of interest.
