import time
import datetime
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.variable import Integer
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.mixed import (
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.optimize import minimize

from chooseopti import choosetime, change_pump_costfun
import config
import os


def initialize_water_network():
    """
    Initializes the water network model from the input file and
    configures pump patterns.

    Returns:
        wntr.network.WaterNetworkModel: The initialized water network model.
    """
    wn = wntr.network.WaterNetworkModel(config.INP_FILE)
    
    # Add custom patterns for pumps that may be missing them
    patterns_to_add = {
        'wyxFixedall0': [0] * 432,
        'pxg6#': [0] * 432,
        'wyxFixedall1': [1] * 432,
    }
    for name, pattern_list in patterns_to_add.items():
        wn.add_pattern(name, pattern_list)

    # Assign patterns to specific pumps
    pump_pattern_map = {
        'wyxFixedall0': ['pxg2', 'pxg4'],
        'pxg6#': ['pxg6'],
        'wyxFixedall1': ['pxg5'],
    }
    for pattern_name, pump_list in pump_pattern_map.items():
        for pump_name in pump_list:
            pump = wn.get_link(pump_name)
            pump.pattern = wn.get_pattern(pattern_name)
            pump.speed_pattern = wn.get_pattern(pattern_name)
            
    # lst=['pxc1','pxc2','pxc3','pxc4','pxc5','pxh1','pxh2','pxh3','pxh4','pxh7','pxh8','pxh9','pxq1','pxq2','pxq3','pxq4','pxq5','pxq1_new','pxq2_nnew','pxq3_new','pxg1','pxg2','pxg3','pxg4','pxg5','pxg6']
    lst=wn.pump_name_list

    # 先补配置中指定的泵
    for pump_name, pattern_name in config.PUMP_PATTERN_MAP.items():
        pump = wn.get_link(pump_name)
        pump.pattern = wn.get_pattern(pattern_name)
        pump.speed_pattern = wn.get_pattern(pattern_name)
        pump.speed_pattern_name = pattern_name

    # 再为所有未补过的泵统一补一个默认 pattern
    for pump_name in lst:
        pump = wn.get_link(pump_name)
        if not hasattr(pump, 'speed_pattern_name') or pump.speed_pattern_name is None:
            pattern_name = 'wyxFixedall0'
            pump.pattern = wn.get_pattern(pattern_name)
            pump.speed_pattern = wn.get_pattern(pattern_name)
            pump.speed_pattern_name = pattern_name
            
    return wn

def get_calculable_pipes(wn):
    """
    Determines the list of pipes to be used in leakage calculations by
    excluding certain pipes from the full list.

    Args:
        wn (wntr.network.WaterNetworkModel): The water network model.

    Returns:
        list: A list of pipe names to be included in calculations.
    """
    all_pipes = set(wn.pipe_name_list)
    
    # Exclude pipes with historically zero or negative pressure
    zero_pressure_pipes = pd.read_csv(config.ZERO_PRESSURE_PIPES_FILE, header=None, delimiter='\t')[0].tolist()
    
    # Combine all excluded pipe lists
    excluded_set = set(config.EXCLUDED_PIPES).union(set(zero_pressure_pipes))
    
    # Return the final list of pipes for calculation
    return list(all_pipes - excluded_set)


class PumpOptimizationProblem(ElementwiseProblem):
    """
    Defines the multi-objective optimization problem for pump scheduling.

    This class sets up the objectives, constraints, and variables for the
    genetic algorithm to solve.
    """
    def __init__(self, simulation_time, headloss_constraints, calculable_pipes):
        self.simulation_time = simulation_time
        self.headloss_constraints = headloss_constraints
        self.calculable_pipes = calculable_pipes
        
        # Define pump variables using settings from the config file
        pump_vars = {
            name: Integer(bounds=details['bounds'])
            for name, details in config.PUMP_VARIABLES.items()
        }

        super().__init__(
            vars=pump_vars,
            n_obj=2,          # Two objective functions (cost and leakage)
            n_ieq_constr=5,   # Five inequality constraints (4 headloss + 1 pressure)
            n_eq_constr=2,    # Two equality constraints
        )
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates the objective functions and constraints for a given set of pump schedules.
        """
        pump_schedule = list(X.values())

        # Constraints for variable speed pumps
        g5 = 1 if 0 < pump_schedule[3] < 7 else 0
        g6 = 1 if 0 < pump_schedule[11] < 7 else 0

        if g5 == 1 and g6 == 1:
            # If constraints are met, run the simulation
            f1, f2, reservoir_levels, pressure_violations = change_pump_costfun(
                wn, 
                self.simulation_time, 
                list(config.PUMP_VARIABLES.keys()), 
                pump_schedule, 
                pipe_cal=self.calculable_pipes, 
                headloss=self.headloss_constraints
            )
            objectives = [f1, f2]
            
            # Unpack reservoir levels for constraints
            c_head, h_head, q_head, g_head = reservoir_levels
            
            # Define inequality constraints based on head loss
            g_inequality = [
                self.headloss_constraints[0] - q_head,  # y_1_2_9_Q
                self.headloss_constraints[1] - h_head,  # y_3_4_5_H
                self.headloss_constraints[2] - c_head,  # y_6_7_C (Restored)
                self.headloss_constraints[3] - g_head,  # y_8_G
                pressure_violations  # Number of nodes with pressure below 20
            ]
        else:
            # Assign high penalty if constraints are violated
            objectives = [10000, 10000]
            g_inequality = [1000, 1000, 1000, 1000, 1000]

        # Set the outputs for the optimizer
        out["F"] = np.array(objectives)
        out["G"] = np.array(g_inequality)
        out["H"] = [1 - g5, 1 - g6] # Equality constraints for VSPs


class OptimizationCallback(Callback):
    """
    A callback to monitor the optimization process and provide feedback.
    """
    def __init__(self):
        super().__init__()
        self.n_eval = 0

    def notify(self, algorithm):
        self.n_eval += 1
        print(f"Evaluation: {self.n_eval}")
        # Terminate if the solution is not improving
        if algorithm.opt is not None and len(algorithm.opt) > 0 and np.any(algorithm.opt[0].F == np.inf):
            print("Terminating due to lack of improvement.")
            algorithm.terminate()
            

def run_optimization(problem):
    """
    Sets up and runs the NSGA-II genetic algorithm.

    Args:
        problem (ElementwiseProblem): The optimization problem to solve.

    Returns:
        pymoo.optimize.Result: The results of the optimization.
    """
    algorithm = NSGA2(
        pop_size=config.POPULATION_SIZE,
            sampling=MixedVariableSampling(), 
            crossover=SBX(prob=1.0, eta=3.0, vtype=int, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=int, repair=RoundingRepair()),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            )

    res = minimize(
        problem,
                    algorithm,
        ('n_gen', config.NUMBER_OF_GENERATIONS),
        seed=config.RANDOM_SEED,
                    verbose=True,
        callback=OptimizationCallback()
    )
    return res


def save_results(results, iteration):
    """
    Saves the Pareto front solutions to CSV files.

    Args:
        results (pymoo.optimize.Result): The optimization results.
        iteration (int): The current iteration number, used for file naming.
    """
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    try:
        pump_status_df = pd.DataFrame([results.X[j] for j in range(results.X.size)])
        f_values_df = pd.DataFrame(results.F, columns=['f1_leakage_cost', 'f2_energy_cost'])

        pump_status_df.to_csv(os.path.join(config.RESULT_PATH, f'{current_time}_iteration_{iteration}_pump_status.csv'))
        f_values_df.to_csv(os.path.join(config.RESULT_PATH, f'{current_time}_iteration_{iteration}_f_values.csv'))
        
        print(f"Results for iteration {iteration} saved successfully.")
    except Exception as e:
        print(f"Could not save results for iteration {iteration}. Error: {e}")


if __name__ == '__main__':
    start_time = time.time()
    
    # Initialize network and pipe lists
    wn = initialize_water_network()
    calculable_pipes = get_calculable_pipes(wn)

    # Load constraints and control nodes from files
    constraints_df = pd.read_excel(config.CONSTRAINTS_FILE)
    head_constraints_data = constraints_df.iloc[:, :9].values
    flow_data = constraints_df.iloc[:, 9].values

    # Main optimization loop
    for i in [5]:  # This loop can be configured for multiple scenarios
        
        target_flow = flow_data[i]
        simulation_time = choosetime(target_value=target_flow)
        print(f"Selected simulation time based on target flow: {simulation_time}")

        # Define head loss constraints for the current scenario
        head_constraints = head_constraints_data[i].tolist()
        head_constraints = [x + 20 for x in head_constraints]
        
        y_1_2_9_Q = max(head_constraints[0], head_constraints[1], head_constraints[8])
        y_3_4_5_H = max(head_constraints[2], head_constraints[4])
        y_6_7_C = max(head_constraints[5], head_constraints[6]) # Restored
        y_8_G = head_constraints[7]
        
        headloss_constraints = [y_1_2_9_Q, y_3_4_5_H, y_6_7_C, y_8_G]
        print(f"Headloss constraints: {headloss_constraints}")

        # Create and run the optimization problem
        problem = PumpOptimizationProblem(
            simulation_time=simulation_time, 
            headloss_constraints=headloss_constraints,
            calculable_pipes=calculable_pipes
        )
        
        res = run_optimization(problem)

        # Save the results
        save_results(res, i)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Optimization finished at: {datetime.datetime.now().strftime('%Y%m%d-%H%M')}")
            



            