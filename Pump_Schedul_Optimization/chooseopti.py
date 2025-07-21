
import pandas as pd
import numpy as np
import wntr
from wntr.network.controls import LinkStatus
import config

def choosetime(target_value):
    """
    Selects the most appropriate simulation time based on the target water demand.

    Args:
        target_value (float): The target water demand value.

    Returns:
        int: The index of the closest time step corresponding to the target demand.
    """
    demand_data = pd.read_excel(config.WATER_DEMAND_FILE)
    demand_list = demand_data['三个水厂模型用水'].tolist()
    
    # Find the index of the value in the list that is closest to the target value
    closest_index = min(range(len(demand_list)), key=lambda i: abs(demand_list[i] - target_value))
    
    return closest_index

def change_pump_costfun(wn, time, pump_names, schedule_list, pipe_cal, headloss):
    """
    Calculates the cost functions (water leakage and electricity) based on a given pump schedule.

    Args:
        wn (wntr.network.WaterNetworkModel): The water network model.
        time (int): The selected time step for the simulation.
        pump_names (list): A list of pump names to be controlled.
        schedule_list (list): The pump schedule (on/off or speed).
        pipe_cal (list): A list of pipes to include in leakage calculations.
        headloss (list): Headloss constraints for the water plants.

    Returns:
        tuple: A tuple containing the water cost, electricity cost, reservoir head levels, and pressure violations.
    """
    time_index = time * 3600
    
    # Adjust speeds for variable speed pumps
    schedule_list[3] *= 0.1
    schedule_list[11] *= 0.1
    schedule_list.insert(21, 0)
    schedule_list.insert(22, 0)

    # Check operational constraints
    g7 = sum(schedule_list[:2])       # Xincun pumps 1 and 2 cannot run simultaneously
    g8 = sum(schedule_list[:5])       # Xincun station can run at most 3 pumps
    g9 = sum(schedule_list[5:7])      # Xinhe pumps 1 and 2 cannot run simultaneously
    g10 = sum(schedule_list[7:9])     # Xinhe pumps 3 and 4 cannot run simultaneously
    g11 = sum(schedule_list[5:12])    # Xinhe station can run at most 4 pumps
    g12 = (sum(schedule_list[12:20]) > 3) or (sum(schedule_list[17:20]) == 2 and sum(schedule_list[12:17]) == 1)
    g13 = sum(schedule_list[-2:])     # Check constraints on the last two pumps
    
    if g7 == 2 or g8 > 3 or g9 >= 2 or g10 >= 2 or g11 > 4 or g12 or g13 == 0:
        return 10000, 10000, [-1, -1, -1, -1], 1000

    # Apply the new pump schedule to the model
    for i, pump_name in enumerate(pump_names):
        pump = wn.get_link(pump_name)
        pump_pattern = pump.speed_pattern_name  # Use only the speed pattern name
        pattern = wn.get_pattern(pump_pattern)  # Assume the pattern always exists
        pattern_new_multiplier = pattern.multipliers
        pattern_new_multiplier[time * 2] = schedule_list[i]
        pattern_new_multiplier[time * 2 + 1] = schedule_list[i]
        pattern.multipliers = pattern_new_multiplier

    # Run the hydraulic simulation
    wn.options.time.pattern_start = time_index
    wn.options.time.duration = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    result = sim.run_sim()

    # Calculate constraints from simulation results
    head_results = result.node['head'].loc[0, :]
    
    # Check if head at water plants meets constraints
    c_max = max(head_results.loc[config.WATER_PLANT_NODES['C']])
    h_max = max(head_results.loc[config.WATER_PLANT_NODES['H']])
    q_max = max(head_results.loc[config.WATER_PLANT_NODES['Q']])
    j_max = head_results.loc[config.WATER_PLANT_NODES['J'][0]]
    all_reservoir_heads = [c_max, h_max, q_max, j_max]

    # --- New Constraint: Check pressure at all nodes ---
    pressure_results = result.node['pressure'].loc[0, :]
    pressure_violations = (pressure_results < 20).sum()


    # --- Objective Function Calculation ---
    # 1. Cost of Water Leakage
    total_leakage = 0
    for pipe_name in pipe_cal:
        pipe = wn.get_link(pipe_name)
        start_node_pressure = result.node['pressure'].loc[0, pipe.start_node_name]
        end_node_pressure = result.node['pressure'].loc[0, pipe.end_node_name]
        avg_pressure = (start_node_pressure + end_node_pressure) / 2
        
        if avg_pressure > 0:
            total_leakage += config.LEAKAGE_COEFFICIENT * pipe.length * (avg_pressure ** config.LEAKAGE_EXPONENT)
            
    cost_water = config.WATER_PRICE_PER_CUBIC_METER * total_leakage

    # 2. Cost of Electricity
    cost_elec = 0
    flowrate_data = result.link['flowrate'].loc[0, :]
    
    for i, pump_name in enumerate(pump_names):
        speed_ratio = schedule_list[i]
        pump_link = wn.get_link(pump_name)
        
        head_increase = head_results.loc[pump_link.end_node_name] - head_results.loc[pump_link.start_node_name]
        flowrate = flowrate_data[pump_name]
        
        # Calculate power and convert to cost
        power_kw = (config.WATER_DENSITY * config.GRAVITY_ACCELERATION * speed_ratio * flowrate * head_increase) / (config.PUMP_EFFICIENCY * 1000)
        cost_elec += config.ELECTRICITY_PRICE_PER_KWH * power_kw

    return cost_water, cost_elec, all_reservoir_heads, pressure_violations


 

