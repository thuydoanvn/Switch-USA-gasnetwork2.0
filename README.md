This version of SWITCH-GAS **endogenizes natural gas production (domestic supply) and demand**. 
In particular, with the 'gas_wells_build' module, the model will decide the optimal number of gas wells 
by type to be drilled and completed in each investment period. Once a gas well is completed, 
it will produce natural gas following a predetermined production curve. 
Demand data is calibrated when running the 'gas_iterative_demand_response' module with 
price elasticity indicated in 'gas_constant_elasticity_demand_system'.
