#!/usr/bin/env python3
"""Evaluation for the paper 'Timing Analysis of Asynchronized Distributed
Cause-Effect Chains' (2021).

It includes (1) local analysis (2) global analysis and (3) plotting of the
results.
"""

#Webserver API
from http.server import BaseHTTPRequestHandler, HTTPServer # python3
import socketserver 
import time

import gc  # garbage collector
import argparse
import math
import numpy as np
import utilities.chain as c
import utilities.communication as comm
import utilities.generator_WATERS as waters
import utilities.generator_UUNIFAST as uunifast
import utilities.transformer as trans
import utilities.event_simulator as es
import utilities.analyzer as a
import utilities.evaluation as eva
import json
import os
import utilities.task as Task
import utilities.chain as Chain
import sys

debug_flag = False  # flag to have breakpoint() when errors occur
unitscale = 1
class end2endServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        BaseHTTPRequestHandler.end_headers(self)
        
    def _set_headers(self):
        self.send_response(200)
        #self.send_header('Content-type', 'text/html')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
    def _set_error_headers(self):
        self.send_response(501)
        self.send_header('Content-type', 'text/html')

        self.end_headers()
    def do_GET(self):
        self._set_headers()
        self.wfile.write("received get request")
        
    def do_POST(self):
        '''Reads post request body'''
        #self._set_headers()
        content_len = int(self.headers.get('content-length'))
        post_body = self.rfile.read(content_len)
        print(post_body)
        system = json.loads(post_body.decode("utf-8"))
        schedule = scheduleLetSynchronise(system)
        if (schedule == None):
            #self.send_response(501, "Scheduler does not support LET parameters")
            self._set_error_headers()
        else:
            self._set_headers()
            self.wfile.write(bytes(json.dumps(schedule),"utf-8"))

    def do_PUT(self):
        self.do_POST();

def main():
    """Main Function."""
    ###
    # Argument Parser
    ###
    parser = argparse.ArgumentParser()

    # which part of code should be executed:
    parser.add_argument("-j", type=int, default=0)
    # utilization in 0 to 100 [percent]:
    parser.add_argument("-u", type=float, default=50)
    # task generation (0: WATERS Benchmark, 1: UUnifast):
    parser.add_argument("-g", type=int, default=0)

    # only for args.j==1:
    # name of the run:
    parser.add_argument("-n", type=int, default=-1)
    # number of task sets to generate:
    parser.add_argument("-r", type=int, default=1)
    
    parser.add_argument("-f", type=str, default="")

    args = parser.parse_args()
    del parser
    if (not os.path.exists('output/1single')):
        os.makedirs('output/1single');
    if (not os.path.exists('output/2interconn')):
        os.makedirs('output/2interconn');
    if (not os.path.exists('output/3plots')):
        os.makedirs('output/3plots');
    if (not os.path.exists('output/LetSynchronise')):
        os.makedirs('output/LetSynchronise');
        
    if args.j == 0: #uses a webserver for scheduling calls
        hostName = "localhost"
        serverPort = 8080
        webServer = HTTPServer((hostName, serverPort), end2endServer)
        print("Server started http://%s:%s" % (hostName, serverPort))

        try:
            webServer.serve_forever()
        except KeyboardInterrupt:
            pass

        webServer.server_close()
        print("Server stopped.")
    elif args.j == 1:
        """Single ECU analysis.

        Required arguments:
        -j1
        -u : utilization [%]
        -g : task generation setting
        -r : number of runs
        -n : name of the run

        Create task sets and cause-effect chains, use TDA, Davare, Duerr, our
        analysis, Kloda, and save the Data
        """
        ###
        # Task set and cause-effect chain generation.
        ###
        print("=Task set and cause-effect chain generation.=")

        try:
            if args.g == 0:
                # WATERS benchmark
                print("WATERS benchmark.")

                # Statistical distribution for task set generation from table 3
                # of WATERS free benchmark paper.
                profile = [0.03 / 0.85, 0.02 / 0.85, 0.02 / 0.85, 0.25 / 0.85,
                           0.25 / 0.85, 0.03 / 0.85, 0.2 / 0.85, 0.01 / 0.85,
                           0.04 / 0.85]
                # Required utilization:
                req_uti = args.u/100.0
                # Maximal difference between required utilization and actual
                # utilization is set to 1 percent:
                threshold = 1.0

                # Create task sets from the generator.
                # Each task is a dictionary.
                print("\tCreate task sets.")
                task_sets_waters = []
                while len(task_sets_waters) < args.r:
                    task_sets_gen = waters.gen_tasksets(
                            1, req_uti, profile, True, threshold/100.0, 4)
                    task_sets_waters.append(task_sets_gen[0])

                # Transform tasks to fit framework structure.
                # Each task is an object of utilities.task.Task.
                trans1 = trans.Transformer("1", task_sets_waters, 10000000)
                task_sets = trans1.transform_tasks(False)

            elif args.g == 1:
                # UUniFast benchmark.
                print("UUniFast benchmark.")

                # Create task sets from the generator.
                print("\tCreate task sets.")

                # The following can be used for task generation with the
                # UUniFast benchmark without predefined periods.

                # # Generate log-uniformly distributed task sets:
                # task_sets_generator = uunifast.gen_tasksets(
                #         5, args.r, 1, 100, args.u, rounded=True)

                # Generate log-uniformly distributed task sets with predefined
                # periods:
                periods = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                # Interval from where the generator pulls log-uniformly.
                min_pull = 1
                max_pull = 2000

                task_sets_uunifast = uunifast.gen_tasksets_pred(
                        50, args.r, min_pull, max_pull, args.u/100.0, periods)

                # Transform tasks to fit framework structure.
                trans2 = trans.Transformer("2", task_sets_uunifast, 10000000)
                task_sets = trans2.transform_tasks(False)

            else:
                print("Choose a benchmark")
                return

            # Create cause effect chains.
            print("\tCreate cause-effect chains")
            ce_chains = waters.gen_ce_chains(task_sets)
            # ce_chains contains one set of cause effect chains for each
            # task set in task_sets.

        except Exception as e:
            print(e)
            print("ERROR: task + ce creation")
            if debug_flag:
                breakpoint()
            else:
                task_sets = []
                ce_chains = []

    

        task_sets, ce_chains = singleECUAnalysis(task_sets, ce_chains)

        ###
        # Save data.
        ###
        print("=Save data.=")

        try:
            np.savez("output/1single/task_set_u="+str(args.u)
                     + "_n=" + str(args.n)
                     + "_g=" + str(args.g) + ".npz", task_sets=task_sets,
                     chains=ce_chains)
        except Exception as e:
            print(e)
            print("ERROR: save")
            if debug_flag:
                breakpoint()
            else:
                return
        
    elif args.j == 2:
        """Interconnected ECU analysis.

        Required arguments:
        -j2
        -u : utilization (for loading)
        -g : task generation setting (for loading)

        Load data, create interconnected chains and then do the analysis by
        Davare, Duerr and Our.
        """

        if args.n == -1:
            print("ERROR: The number of runs -n is not specified.")
            return

        # Variables.
        utilization = args.u
        gen_setting = args.g
        num_runs = args.n
        number_interconn_ce_chains = 10000

        try:
            ###
            # Load data.
            ###
            print("=Load data.=")
            chains_single_ECU = []
            for i in range(num_runs):
                name_of_the_run = str(i)
                data = np.load(
                        "output/1single/task_set_u=" + str(utilization)
                        + "_n=" + name_of_the_run
                        + "_g=" + str(gen_setting)
                        + ".npz", allow_pickle=True)
                for chain_set in data.f.chains:
                    for chain in chain_set:
                        chains_single_ECU.append(chain)

                # Close data file and run the garbage collector.
                data.close()
                del data
                gc.collect()
        except Exception as e:
            print(e)
            print("ERROR: inputs from single are missing")
            if debug_flag:
                breakpoint()
            else:
                return

        ###
        # Interconnected cause-effect chain generation.
        ###
        print("=Interconnected cause-effect chain generation.=")
        chains_inter = []
        for j in range(0, number_interconn_ce_chains):
            chain_all = []  # sequence of all tasks (from chains + comm tasks)
            i_chain_all = []  # sequence of chains and comm_tasks

            # Generate communication tasks.
            com_tasks = comm.generate_communication_taskset(20, 10, 1000, True)

            # Fill chain_all and i_chain_all.
            k = 0
            for chain in list(np.random.choice(
                    chains_single_ECU, 5, replace=False)):  # randomly choose 5
                i_chain_all.append(chain)
                for task in chain.chain:
                    chain_all.append(task)
                if k < 4:  # communication tasks are only added in between
                    chain_all.append(com_tasks[k])
                    i_chain_all.append(com_tasks[k])
                k += 1

            chains_inter.append(c.CauseEffectChain(0, chain_all, i_chain_all))

            # End user notification
            if j % 100 == 0:
                print("\t", j)

        ###
        # Analyses (Davare, Duerr, Our).
        # Kloda is not included, since it is only for synchronized clocks.
        ###
        print("=Analyses (Davare, Duerr, Our).=")
        analyzer = a.Analyzer("0")

        print("Test: Davare.")
        analyzer.davare([chains_inter])

        print("Test: Duerr.")
        analyzer.reaction_duerr([chains_inter])
        analyzer.age_duerr([chains_inter])

        print("Test: Our.")
        # Our test can only be used when the single processor tests are already
        # done.
        analyzer.max_age_inter_our(chains_inter, reduced=True)
        analyzer.reaction_inter_our(chains_inter)

        ###
        # Save data.
        ###
        print("=Save data.=")
        np.savez(
                "./output/2interconn/chains_" + "u=" + str(utilization)
                + "_g=" + str(gen_setting) + ".npz",
                chains_inter=chains_inter, chains_single_ECU=chains_single_ECU)

    elif args.j == 3:
        """Evaluation.

        Required arguments:
        -j3
        -g : task generation setting (for loading)
        """
        # Variables.
        gen_setting = args.g
        utilizations = [50.0, 60.0, 70.0, 80.0, 90.0]

        try:
            ###
            # Load data.
            ###
            print("=Load data.=")
            chains_single_ECU = []
            chains_inter = []
            for ut in utilizations:
                data = np.load(
                        "output/2interconn/chains_" + "u=" + str(ut)
                        + "_g=" + str(args.g) + ".npz", allow_pickle=True)

                # Single ECU.
                for chain in data.f.chains_single_ECU:
                    chains_single_ECU.append(chain)

                # Interconnected.
                for chain in data.f.chains_inter:
                    chains_inter.append(chain)

                # Close data file and run the garbage collector.
                data.close()
                del data
                gc.collect()
        except Exception as e:
            print(e)
            print("ERROR: inputs for plotter are missing")
            if debug_flag:
                breakpoint()
            else:
                return

        ###
        # Draw plots.
        ###
        print("=Draw plots.=")

        myeva = eva.Evaluation()

        # Single ECU Plot.
        myeva.davare_boxplot_age(
                chains_single_ECU,
                "output/3plots/davare_single_ecu_age"
                + "_g=" + str(args.g) + ".pdf",
                xaxis_label="", ylabel="Latency reduction [%]")
        myeva.davare_boxplot_reaction(
                chains_single_ECU,
                "output/3plots/davare_single_ecu_reaction"
                + "_g=" + str(args.g) + ".pdf",
                xaxis_label="", ylabel="Latency reduction [%]")

        # Interconnected ECU Plot.
        myeva.davare_boxplot_age_interconnected(
                chains_inter,
                "output/3plots/davare_interconnected_age"
                + "_g=" + str(args.g) + ".pdf",
                xaxis_label="", ylabel="Latency reduction [%]")
        myeva.davare_boxplot_reaction_interconnected(
                chains_inter,
                "output/3plots/davare_interconnected_reaction"
                + "_g=" + str(args.g) + ".pdf",
                xaxis_label="", ylabel="Latency reduction [%]")

        # # Heatmap.
        # myeva.heatmap_improvement_disorder_age(
        #         chains_single_ECU,
        #         "output/3plots/heatmap" + "_our_age"
        #         + "_g=" + str(args.g) + ".pdf",
        #         yaxis_label="")
        # myeva.heatmap_improvement_disorder_react(
        #         chains_single_ECU,
        #         "output/3plots/heatmap" + "_our_react"
        #         + "_g=" + str(args.g) + ".pdf",
        #         yaxis_label="")
    elif args.j == 4:
        """Evaluation.

        Required arguments:
        -j3
        -g : task generation setting (for loading)
        -u : ulilisation
        -n
        """
        # Variables.
        #gen_setting = args.g
        #utilizations = [50.0, 60.0, 70.0, 80.0, 90.0]
        
        
        
        try:
            ###
            # Load data.
            ###
            print("=Load data.=")
            #chains_single_ECU = []
            #chains_inter = []
            #python main.py -j4 -g1 -u50 -n0
            #python main.py -j4 -g0 -u50 -n0
            data = np.load(
                    "output/1single/task_set_" + "u=" + str(args.u)
                    + "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz", allow_pickle=True)

            print(data.f)
            task_sets = data.f.task_sets
            chains = data.f.chains
            
            system = export_letsSyncrhonise_json(task_sets, chains, None)
            with open('output/LetSynchronise/system.json', 'w') as outfile:
                json.dump(system, outfile, indent=4)
                # Interconnected.
            #for chain in data.f.chains_inter:
            #    chains_inter.append(chain)

            # Close data file and run the garbage collector.
            data.close()
            del data
            gc.collect()
        except Exception as e:
            print(e)
            if debug_flag:
                breakpoint()
            else:
                return
    elif args.j == 5:
        try:
            ###
            # Load data.
            ###
            print("=Load data.=")
            #chains_single_ECU = []
            #chains_inter = []
            #python main.py -j5 -g1 -u50 -n0
            #python main.py -j5 -g0 -u50 -n0
            data = np.load(
                    "output/1single/task_set_" + "u=" + str(args.u)
                    + "_n=" + str(args.n) + "_g=" + str(args.g) + ".npz", allow_pickle=True)

            print(data.f)
            #tasks_single_ECU = [] 
            #chains_single_ECU = []
            #for chain_set in data.f.chains:
            #    for chain in chain_set:
            #        chain.davare = 0  # Davare
            #        chain.duerr_age = 0  # Duerr max data age
            #        chain.duerr_react = 0  # Duerr max reaction time
            #        chain.our_age = 0  # Our max data age
            #        chain.our_react = 0  # Our max reaction time
            #        chain.our_red_age = 0  # Our reduced max data age
            #        chain.inter_our_age = 0  # Our max data age for interconn
            #        chain.inter_our_red_age = 0  # Our reduced max data age for interconn
            #        chain.inter_our_react = 0  # Our max reaction time for interconn
            #        chain.kloda = 0  # Kloda
            #for task_set in data.f.task_sets:
            #    for task in task_set:

            #        tasks_single_ECU.append(task)
            #        chains_single_ECU.append(chain)
            task_sets = data.f.task_sets
            chains = data.f.chains

            #for set in task_sets:
            #    for s in set:
            #        s.rt = 0
            #        print(s)
            #for c in chains:
            #    for s in c:
            #        print(s)
            #for task_set in data.f.task_sets:
            #    for task in task_set:
            #        tasks_single_ECU.append(task)

            print("===Begin analysis===")
            #print(task_sets)
            #print(chains)
            relink_chains(task_sets, chains)
            #ce_chains = waters.gen_ce_chains(task_sets)
            task_sets, chains = singleECUAnalysis(task_sets, chains)
            
            # Close data file and run the garbage collector.
            data.close()
            del data
            gc.collect()
        except Exception as e:
            print(e)
            if debug_flag:
                breakpoint()
            else:
                return
    elif args.j == 6:
        
        if (len(args.f) == 0):
            print("Please specify input LetSynchronise json")
            return 0;
               
        #f = open('output/LetSynchronise/system.json')
        f = open(args.f)
        system = json.load(f)
        scheduleLetSynchronise(system)
def getDependencyInstances(system, name):
    for d in system["DependencyInstancesStore"]: 
        if d['name'] == name:
            return d;
    return None  
    
def getNextDependencyInstance(system, dependencyInstName, afterTime):
    instances = getDependencyInstances(system, dependencyInstName)
    for inst in instances['value']:
        if (inst['sendEvent']['timestamp'] >=  afterTime):
            return inst
    return None  
    
def scheduleLetSynchronise(system):
    
    #"ConstraintStore" , "DependencyStore", "EventChainStore", "SystemInputStore", "SystemOutputStore", "TaskStore" 
    task_set = []
    chains = []
    task_id_map = {}
    id_task_map = {} #to get original information back
    id_counter = 1 # reserved zero for system
    task_gcd_period = -1;

    #Reject task where activation offset is non zero 
    
    for t in system['TaskStore']:
        if (t['activationOffset'] != 0):
            print("\n\nError this tool does not suppor tasks with activation offset.\n\n")
            return None;
        #task_set.append(Task.Task(task_id=id_counter, task_phase=int(t['initialOffset'] * unitscale), task_bcet=int(t['bcet']), task_wcet=int(t['wcet']), task_period=int(t['period']*unitscale), task_deadline=int(t['duration']*unitscale), priority=t['priority'], message=t['message']))
        task_set.append(Task.Task(task_id=id_counter, task_phase=int(t['initialOffset'] * unitscale), task_bcet=int(t['bcet']* unitscale), task_wcet=int(t['wcet']* unitscale), task_period=int(t['period']*unitscale), task_deadline=int(t['duration']*unitscale), priority=id_counter, message=False))
        task_id_map[str(t['name'])] = id_counter
        id_task_map[str(id_counter)] = t
        id_counter = id_counter + 1
        if (task_gcd_period == -1):
            task_gcd_period = int(t['period']*unitscale)
        else:
            task_gcd_period = math.gcd(task_gcd_period,int(t['period']*unitscale))
    
    
    #Create System Task for LetSynchronise
    id_counter = 0
    task_id_map["__system"] = id_counter
    #wcet is smallest non-zero value
    task_set.insert(0,Task.Task(task_id=id_counter, task_phase=0, task_bcet=0, task_wcet=sys.float_info.min, task_period=task_gcd_period, task_deadline=task_gcd_period, priority=id_counter, message=False))
    id_task_map[str(id_counter)] = {"name":"__system"}
    
    print (id_task_map)
    
    id_counter = 0
    for c in system['EventChainStore']:
        chain = []
        successor = c.get('successor')
        #print(c)
        #print("-------------------------")
        #print(c.get('segment').get('source').get('task'))
        #print(task_id_map.get(c.get('segment').get('source').get('task')))
        chain.append(task_set[task_id_map.get(c.get('segment').get('source').get('task'))])
        chain.append(task_set[task_id_map.get(c.get('segment').get('destination').get('task'))])
        while(successor != None):
            chain.append(task_set[task_id_map.get(successor.get('segment').get('destination').get('task'))])
            successor = successor.get('successor')
        chains.append(Chain.CauseEffectChain(id = id_counter, chain=chain, interconnected=[]))
        id_counter = id_counter + 1

    for t in task_set:
        print(t)
    print("===Begin analysis===")
    task_sets = [task_set] #single set
    ce_chains = [chains] #single chain set

    #task_sets, chains = singleECUAnalysis(task_sets, ce_chains)
    schedules, task_sets, chains = scheduleSingleECUAnalysis(task_sets, ce_chains)
    

    
    print("---debug----")
    print(schedules)
    
    fo = open("output/schedule.txt", "w")
    result = schedules[0].e2e_result()
    
    #Dependency Instance
    #"name": "alpha",
    #"value": [
    #{
    #  "instance": 0,
    #  "receiveEvent": {
    #    "task": "task-a",
    #    "port": "in",
    #    "taskInstance": 0,
    #    "timestamp": 1
    #  },
    #  "sendEvent": {
    #    "task": "__system",
    #    "port": "SystemInput",
    #    "taskInstance": 0,
    #    "timestamp": 1
    #  }
    #},
    #...
    #]
    
    #{
    #  "segment": {
    #    "name": "alpha",
    #    "instance": 0,
    #    "receiveEvent": {
    #      "task": "task-a",
    #      "port": "in",
    #      "taskInstance": 0,
    #      "timestamp": 1
    #    },
    #    "sendEvent": {
    #      "task": "__system",
    #      "port": "SystemInput",
    #      "taskInstance": 0,
    #      "timestamp": 1
    #    }
    #  },
    #  "name": "EventChain1-0",
    #  "successor": {
    #    "segment": {
    #      "name": "beta",
    #      "instance": 1,
    #      "receiveEvent": {
    #        "task": "task-c",
    #        "port": "in1",
    #        "taskInstance": 1,
    #        "timestamp": 3
    #      },
    #      "sendEvent": {
    #        "task": "task-a",
    #        "port": "out",
    #        "taskInstance": 0,
    #        "timestamp": 3
    #      }
    #    },
    #    "successor": {
    #      "segment": {
    #        "name": "delta",
    #        "instance": 1,
    #        "receiveEvent": {
    #          "task": "__system",
    #          "port": "SystemOutput",
    #          "taskInstance": 1,
    #          "timestamp": 4
    #        },
    #        "sendEvent": {
    #          "task": "task-c",
    #          "port": "out",
    #          "taskInstance": 1,
    #          "timestamp": 4
    #        }
    #      }
    #    }
    #  }
    #}
    
    #Task Instance
    
    #"name": "task-a",
    #"initialOffset": 0,
    #"value": [
    #{
    #  "instance": 0,
    #  "periodStartTime": 0,
    #  "letStartTime": 1,
    #  "letEndTime": 3,
    #  "periodEndTime": 3,
    #  "executionTime": 0.9478027315786561,
    #  "executionIntervals": [
    #    {
    #      "startTime": 1,
    #      "endTime": 1.473901365789328
    #    },
    #    {
    #      "startTime": 2.5260986342106717,
    #      "endTime": 3
    #    }
    #  ]
    #},
    #...
    #]
    
   
    
    #export schedule
    schedule = {
        "DependencyInstancesStore" : [], 
        "EventChainInstanceStore" : [],
        "TaskInstancesStore" : []
        }
        
    for t in task_set:
        parameters = result.get(t)
        fo.write("Task: "+t.id+"\n")
        taskInstancesJson = {
            "name" : id_task_map[str(t.id)].get("name"),
            "initialOffset" : 0,
        }
        instances = []
        for i in range(0, len(parameters)):
            fo.write("j"+str(i)+" - " + "start: "+str(parameters[i][0]) + " end: " +str(parameters[i][1])+"\n")
            starttime = parameters[i][0]
            if (starttime == sys.float_info.min):
                starttime = 0
            endtime = parameters[i][1]
            taskInstance = {
                "instance" : i,
                "periodStartTime" : (i * t.period+t.phase)/unitscale,
                #"letStartTime" : starttime/unitscale,
                #"letEndTime" : endtime/unitscale,
                "letStartTime" : (i * t.period+t.phase)/unitscale,
                "letEndTime" : (i * t.period+t.phase+t.deadline)/unitscale,
                "periodEndTime" : ((i+1) * t.period+t.phase)/unitscale,
                "executionTime": (endtime-starttime)/unitscale,
                "executionIntervals": [ {
                    "startTime": starttime/unitscale,
                    "endTime": endtime/unitscale
                } ]
            }
            instances.append(taskInstance)
        taskInstancesJson["value"] = instances
        if (id_task_map[str(t.id)].get("name") != "__system"):
            schedule["TaskInstancesStore"].append(taskInstancesJson)
        
    fo.close()
    #export system
    export = export_letsSyncrhonise_json(task_sets, chains, id_task_map)
    
    #compute dependencyInstance 
    export['DependencyInstancesStore'] = []
    for dep in system['DependencyStore']:
        dependencyInstance = {}
        dependencyInstance['name'] = dep['name'];
        instances = []
        for destTaskInsts in schedule["TaskInstancesStore"]:
            if destTaskInsts['name'] == dep['destination']['task']:
                for destTaskInst in destTaskInsts["value"]:
                    for srcTaskInsts in schedule["TaskInstancesStore"]:
                        if dep['source']['task'] == "__system": 
                            instance = {
                               "instance":destTaskInst['instance'],
                               "receiveEvent":{
                                  "task":dep['destination']['task'],
                                  "port":dep['destination']['port'],
                                  "taskInstance":destTaskInst['instance'],
                                  "timestamp":destTaskInst['letStartTime']
                               },
                               "sendEvent":{
                                  "task":dep['source']['task'],
                                  "port":dep['source']['port'],
                                  "taskInstance":destTaskInst['instance'],
                                  "timestamp":destTaskInst['letStartTime'] #system have same instance and time
                               }
                            }
                            instances.append(instance);
                        if srcTaskInsts['name'] == dep['source']['task']:
                            closestSrcInst = srcTaskInsts["value"][0]
                            minTimeDiff = float('inf') 
                            for srcTaskInst in srcTaskInsts["value"]:
                                if (destTaskInst['letStartTime'] - srcTaskInst['letEndTime'] >= 0): #has to be positive
                                    if (minTimeDiff < destTaskInst['letStartTime'] - srcTaskInst['letEndTime']):
                                        minTimeDiff = destTaskInst['letStartTime'] - srcTaskInst['letEndTime']
                                        closestSrcInst = srcTaskInst
                                    
                            instance = {
                               "instance":destTaskInst['instance'],
                               "receiveEvent":{
                                  "task":dep['destination']['task'],
                                  "port":dep['destination']['port'],
                                  "taskInstance":destTaskInst['instance'],
                                  "timestamp":destTaskInst['letStartTime']
                               },
                               "sendEvent":{
                                  "task":dep['source']['task'],
                                  "port":dep['source']['port'],
                                  "taskInstance":closestSrcInst['instance'],
                                  "timestamp":closestSrcInst['letStartTime']
                               }
                            }
                            
                            instances.append(instance);
        dependencyInstance['value'] = instances;
        export['DependencyInstancesStore'].append(dependencyInstance)
    print("------------------------------------------------------------------------");
    print(export["DependencyInstancesStore"]);
    print("------------------------------------------------------------------------");
    #compute eventChainInstance
    export['EventChainInstanceStore'] = []
    for c in system['EventChainStore']:
        
        dependencyInstances = getDependencyInstances(export, c["segment"]["name"])
        i = 0
        for dependencyInst in dependencyInstances['value']:
            successor = c.get('successor')
            
            complete = True
            evtChainInst = {}
            evtChainInst["name"] = c["name"]+"-"+str(i)
            evtChainInst["segment"] = dependencyInst
            evtChainInst["segment"]["name"] = dependencyInstances["name"]
            #current successor
            current = evtChainInst
            
            while(successor != None):
                #find closest successor instance
                afterTime = evtChainInst["segment"]["receiveEvent"]["timestamp"]
                for t in system['TaskStore']:
                    if t['name'] == current["segment"]["receiveEvent"]["task"]:
                        afterTime = afterTime + t["duration"]
                result = getNextDependencyInstance(export, successor["segment"]["name"], evtChainInst["segment"]["receiveEvent"]["timestamp"])
                if (result == None):
                    #print ("XX")
                    #exit(0)
                    complete = False
                    break
                    
                #assign as successor
                current["successor"] = {}
                current["successor"]["segment"] = result
                current["successor"]["segment"]["name"] = successor["segment"]["name"]
                current = current["successor"]
                successor = successor.get('successor')
            if complete:    
                export['EventChainInstanceStore'].append(evtChainInst)
            i = i + 1
        
        

    #print(export['ConstraintStore'])
    #print("Xxxxxxx");
    #exit(0);
    
    #As export does a backward conversion from end-to-end format, it loses information so its best to use original information.
    export['SystemInputStore'] = system['SystemInputStore']
    export['SystemOutputStore'] = system['SystemOutputStore']
    export['TaskStore'] = system['TaskStore']
    schedule.update(export)
    schedule['DependencyStore'] = system['DependencyStore'] #restore missing information since end-to-end does not have this information
    schedule['EventChainStore'] = system['EventChainStore'] #restore missing information since end-to-end does not have this information
    
    
    
    with open('output/LetSynchronise/system-schedule.json', 'w') as outfile:
        json.dump(schedule, outfile, indent=4)
    print("RESULTS!!")
    schedules[0].tableReport()
    print("RESULTS!!")
    print(result)
    print("total miss rate: "+str(schedules[0].totalMissRate()))
    #print(schedule['ConstraintStore'])
    
    #
    return schedule
        

def export_letsSyncrhonise_json(task_sets, chains, id_task_map):
    
    #LetSynchronise data structure
    system = {
        "ConstraintStore" : [], 
        "DependencyStore" : [],
        "EventChainStore" : [],
        "SystemInputStore" : [],
        "SystemOutputStore" : [],
        "TaskStore" : [],
        }
    # Single ECU.
    for idxx in range(len(task_sets)):
        print("--------------------------------------- " + str(idxx))
        for task in task_sets[idxx]:
            print(task)
            #self.id 
            #self.phase 
            #self.bcet 
            #self.wcet 
            #self.period 
            #self.deadline 
            #self.priority 
            #self.message 
            if (id_task_map == None):
                l_task = {
                    "name": "task"+str(task.id),
                     "initialOffset":(task.phase/unitscale),
                     "activationOffset":0,
                     "duration":(task.deadline/unitscale),
                     "period":(task.period/unitscale),
                     "inputs":[
                        "in"
                     ],
                     "outputs":[
                        "out"
                     ],
                    "wcet" : (task.wcet/unitscale),
                    "bcet" : (task.bcet/unitscale),
                    "acet" : ((task.wcet-task.bcet)/2+task.bcet)/unitscale, # assume averge is in the middle
                    "distribution":"Uniform" #assume uniform distribution
                    #"priority" : task.priority, #not currently accepted by LetSynchronise
                    #"message" : task.message  #not currently accepted by LetSynchronise
                }
            else:
                if (id_task_map[str(task.id)].get("name") == "__system"):
                    continue; # skip system task
                l_task = {
                    "name": id_task_map[str(task.id)].get("name"),
                     "initialOffset":(task.phase/unitscale),
                     "activationOffset":0,
                     "duration":(task.deadline/unitscale),
                     "period":(task.period/unitscale),
                     "inputs":[
                        "in" #todo: need to fix using a port map
                     ],
                     "outputs":[
                        "out" #todo: need to fix
                     ],
                    "wcet" : (task.wcet/unitscale),
                    "bcet" : (task.bcet/unitscale),
                    "acet" : id_task_map[str(task.id)].get("acet"), # get the acet
                    "distribution":"Uniform" #assume uniform distribution
                    #"priority" : task.priority, #not currently accepted by LetSynchronise
                    #"message" : task.message  #not currently accepted by LetSynchronise
                }
            system["TaskStore"].append(l_task)
        for chain in chains[idxx]:
            l_chain = {};
            l_chain_last = {};
            print("chain: "+str(chain.id))
            first = True
            second = True
            previousTask = None #assuming chain is by order
            for task in chain.chain:
                print("T:"+str(task.id))
                if first:
                    previousTask = task;
                    first = False
                    continue;
                else:
                    dependencyName = "dep_"+str(chain.id)+"_"+str(previousTask.id)+"_"+str(task.id) #create dependencies again as end-to-end does not have this information
                    if (id_task_map == None):
                        dependencySourceTask = "task"+str(previousTask.id)
                    else:
                        dependencySourceTask =  id_task_map[str(previousTask.id)].get("name")
                    
                    dependencySourcePort = "out"
                    
                    if (id_task_map == None):
                        dependencyDestTask = "task"+str(task.id)
                    else:
                        dependencyDestTask = id_task_map[str(task.id)].get("name")
                    dependencyDestPort = "in"
                    l_dependency = {
                         "name":dependencyName,
                         "source":{
                            "task":dependencySourceTask,
                            "port":dependencySourcePort
                         },
                         "destination":{
                            "task":dependencyDestTask,
                            "port":dependencyDestPort
                         }
                    }
                    system["DependencyStore"].append(l_dependency) #create missing information from model
                    
                    #event chain
                    if second:
                        second = False
                        l_chain_last = {"name" : "chain_"+str(chain.id),
                                   "segment":{
                                        "name":dependencyName,
                                        "source":{
                                           "task":dependencySourceTask,
                                           "port":dependencySourcePort
                                        },
                                        "destination":{
                                           "task":dependencyDestTask,
                                           "port":dependencyDestPort
                                        }
                                    }
                                  }
                        l_chain = l_chain_last
                    else:
                        l_successor = {"segment":{
                                        "name":dependencyName,
                                        "source":{
                                           "task":dependencySourceTask,
                                           "port":dependencySourcePort
                                        },
                                        "destination":{
                                           "task":dependencyDestTask,
                                           "port":dependencyDestPort
                                        }
                                    }}
                        
                        l_chain_last["successor"] = l_successor
                        print(l_chain)
                        print(l_chain_last)
                        l_chain_last = l_chain_last["successor"]
                        
                    previousTask = task;
            system["EventChainStore"].append(l_chain)
            l_constraint = {
                "name" : "chain_"+str(chain.id)+"reaction_time",
                "eventChain" : "chain_"+str(chain.id) ,
                "relation" : "<=",
                "time": chain.our_react/unitscale
                
                
            }
            system["ConstraintStore"].append(l_constraint);
    
    
    return system;
        
def relink_chains(task_sets, chains):
    ce_chains = []
    for idxx in range(len(task_sets)):
        for c in chains[idxx]:
            #print (c.chain)
            for i in range(len(c.chain)):
                for t in task_sets[idxx]:
                    if (c.chain[i].id == t.id):
                        c.chain[i] = t
                        break
            #interconnect not yet taken care of as focused on single ECU

    return ce_chains
    
def scheduleSingleECUAnalysis(task_sets, ce_chains):
    ###
    # First analyses (TDA, Davare, Duerr).
    ###
    print("=First analyses (TDA, Davare, Duerr).=")
    analyzer = a.Analyzer("0")

    try:
        # TDA for each task set.
        print("TDA.")
        for idxx in range(len(task_sets)):
            try:
                # TDA.
                i = 1
                for task in task_sets[idxx]:
                    # Prevent WCET = 0 since the scheduler can
                    # not handle this yet. This case can occur due to
                    # rounding with the transformer.
                    if task.wcet == 0:
                        raise ValueError("WCET == 0")
                    task.rt = analyzer.tda(task, task_sets[idxx][:(i - 1)])
                    if task.rt > task.deadline:
                        raise ValueError(
                                    "TDA Result: WCRT bigger than deadline!")
                    i += 1
            except ValueError:
                # If TDA fails, remove task and chain set and continue.
                task_sets.remove(task_sets[idxx])
                ce_chains.remove(ce_chains[idxx])
                continue
             
        # End-to-End Analyses. 
        # These results are used to for schedule computation
        res = analyzer.davare(ce_chains)
        res = analyzer.reaction_duerr(ce_chains)
        res = analyzer.age_duerr(ce_chains)


        ###
        # Second analyses (Simulation, Our, Kloda).
        ###
        print("=Second analyses (Simulation, Our, Kloda).=")
        i = 0  # task set counter
        schedules = []
        for task_set in task_sets:
            print("=Task set ", i+1)

            # Skip if there is no corresponding cause-effect chain.
            if len(ce_chains[i]) == 0:
                continue

            # Event-based simulation.
            print("Simulation.")

            simulator = es.eventSimulator(task_set)
            # Determination of the variables used to compute the stop
            # condition of the simulation
            max_e2e_latency = max(ce_chains[i], key=lambda chain:
                                      chain.davare).davare
            max_phase = max(task_set, key=lambda task: task.phase).phase
            max_period = max(task_set, key=lambda task: task.period).period
            hyper_period = analyzer.determine_hyper_period(task_set)

            sched_interval = (
                        2 * hyper_period + max_phase  # interval from paper
                        + max_e2e_latency  # upper bound job chain length
                        + max_period)  # for convenience

            # Information for end user.
            print("\tNumber of tasks: ", len(task_set))
            print("\tHyperperiod: ", hyper_period)
            number_of_jobs = 0
            for task in task_set:
                number_of_jobs += sched_interval/task.period
            print("\tNumber of jobs to schedule: ",
                      "%.2f" % number_of_jobs)

            # Stop condition: Number of jobs of lowest priority task.
            simulator.dispatcher(
                        int(math.ceil(sched_interval/task_set[-1].period)))
            
            # Simulation without early completion.
            schedule = simulator.e2e_result()
            schedules.append(simulator) #output raw simulator

            # Analyses.
            for chain in ce_chains[i]:
                print("Test: Our Data Age.")
                analyzer.max_age_our(schedule, task_set, chain, max_phase,
                                     hyper_period, reduced=False)
                analyzer.max_age_our(schedule, task_set, chain, max_phase,
                                     hyper_period, reduced=True)

                print("Test: Our Reaction Time.")
                analyzer.reaction_our(schedule, task_set, chain, max_phase,
                                      hyper_period)

                # Kloda analysis, assuming synchronous releases.
                print("Test: Kloda.")
                analyzer.kloda(chain, hyper_period)

                # Test.
                if chain.kloda < chain.our_react:
                    if debug_flag:
                        breakpoint()
                    else:
                        raise ValueError(
                                ".kloda is shorter than .our_react")
            i += 1

        

        return schedules,task_sets,ce_chains
    except Exception as e:
        print(e)
        print("ERROR: analysis")
        if debug_flag:
            breakpoint()
        else:
            task_sets = []
            ce_chains = []
            
    #1 task set will return 1 schedule.
    return schedules,task_sets,ce_chains
    
def singleECUAnalysis(task_sets, ce_chains):
    ###
    # First analyses (TDA, Davare, Duerr).
    ###
    print("=First analyses (TDA, Davare, Duerr).=")
    analyzer = a.Analyzer("0")

    try:
        # TDA for each task set.
        print("TDA.")
        for idxx in range(len(task_sets)):
            try:
                # TDA.
                i = 1
                for task in task_sets[idxx]:
                    # Prevent WCET = 0 since the scheduler can
                    # not handle this yet. This case can occur due to
                    # rounding with the transformer.
                    if task.wcet == 0:
                        raise ValueError("WCET == 0")
                    task.rt = analyzer.tda(task, task_sets[idxx][:(i - 1)])
                    if task.rt > task.deadline:
                        raise ValueError(
                                    "TDA Result: WCRT bigger than deadline!")
                    i += 1
            except ValueError:
                # If TDA fails, remove task and chain set and continue.
                task_sets.remove(task_sets[idxx])
                ce_chains.remove(ce_chains[idxx])
                continue
             
        # End-to-End Analyses.
        print("Test: Davare.")
        res = analyzer.davare(ce_chains)
        print("Davare End-to-End: "+ str(res))


        print("Test: Duerr Reaction Time.")
        res = analyzer.reaction_duerr(ce_chains)
        print("Duerr Reaction Time: "+ str(res))

        print("Test: Duerr Data Age.")
        res = analyzer.age_duerr(ce_chains)
        print("Duerr Data Age: "+ str(res))

        ###
        # Second analyses (Simulation, Our, Kloda).
        ###
        print("=Second analyses (Simulation, Our, Kloda).=")
        i = 0  # task set counter
        schedules = []
        for task_set in task_sets:
            print("=Task set ", i+1)

            # Skip if there is no corresponding cause-effect chain.
            if len(ce_chains[i]) == 0:
                continue

            # Event-based simulation.
            print("Simulation.")

            simulator = es.eventSimulator(task_set)
            # Determination of the variables used to compute the stop
            # condition of the simulation
            max_e2e_latency = max(ce_chains[i], key=lambda chain:
                                      chain.davare).davare
            max_phase = max(task_set, key=lambda task: task.phase).phase
            max_period = max(task_set, key=lambda task: task.period).period
            hyper_period = analyzer.determine_hyper_period(task_set)

            sched_interval = (
                        2 * hyper_period + max_phase  # interval from paper
                        + max_e2e_latency  # upper bound job chain length
                        + max_period)  # for convenience

            # Information for end user.
            print("\tNumber of tasks: ", len(task_set))
            print("\tHyperperiod: ", hyper_period)
            number_of_jobs = 0
            for task in task_set:
                number_of_jobs += sched_interval/task.period
            print("\tNumber of jobs to schedule: ",
                      "%.2f" % number_of_jobs)

            # Stop condition: Number of jobs of lowest priority task.
            simulator.dispatcher(
                        int(math.ceil(sched_interval/task_set[-1].period)))

            # Simulation without early completion.
            schedule = simulator.e2e_result()
            schedules.append(schedule) #output raw simulator

            # Analyses.
            for chain in ce_chains[i]:
                print("Test: Our Data Age.")
                res = analyzer.max_age_our(schedule, task_set, chain, max_phase,
                                         hyper_period, reduced=False)
                print("Our Data Age One:" + str(res))
                res = analyzer.max_age_our(schedule, task_set, chain, max_phase,
                                         hyper_period, reduced=True)
                print("Our Data Age Two:" + str(res))
                print("Test: Our Reaction Time.")
                res = analyzer.reaction_our(schedule, task_set, chain, max_phase,
                                          hyper_period)
                print("Our Reaction Time:" + str(res))
                    # Kloda analysis, assuming synchronous releases.
                print("Test: Kloda.")
                res = analyzer.kloda(chain, hyper_period)
                print("Kloda Analysis:" + str(res))
                    # Test.
                if chain.kloda < chain.our_react:
                    if debug_flag:
                        breakpoint()
                    else:
                        raise ValueError(
                                    ".kloda is shorter than .our_react")
            i += 1
    except Exception as e:
        print(e)
        print("ERROR: analysis")
        if debug_flag:
            breakpoint()
        else:
            task_sets = []
            ce_chains = []
    return task_sets,ce_chains
if __name__ == '__main__':
    main()
