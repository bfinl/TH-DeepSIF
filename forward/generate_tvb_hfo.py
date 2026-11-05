"""Adapted from the DeepSIF neural mass model pipeline (generate_tvb_data.py),
https://github.com/bfinl/DeepSIF -- Sun et al., Proc Natl Acad Sci 2022,
doi:10.1073/pnas.2201128119
Original code Copyright (c) 2022, authors of DeepSIF, released under the Clear
BSD License. Modified here to simulate high-frequency oscillations.

Source dynamics use the Jansen-Rit neural mass model (Jansen & Rit, Biol Cybern
1995) via The Virtual Brain (Sanz Leon et al., Front Neuroinform 2013).
See ../NOTICE and forward/README.md.
"""

import pywt
import matplotlib.pyplot as plt
import numpy
from tvb.simulator.lab import *
from scipy import signal
from scipy.io import savemat
import random
import argparse
import os
import time

def highpass(data: numpy.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def main(region_id):

    if not os.path.isdir('hfo_source/raw_nmm_10012024/region{}'.format(region_id)):
        os.mkdir('hfo_source/raw_nmm_10012024/region{}'.format(region_id))

    """ TVB Simulation to generate raw source space dynamics, unit in mV, and ms
    :param region_id: int; source region id, with parameters generating interictal spike activity
    """
    print("check")
    conn = connectivity.Connectivity.from_file(source_file=os.getcwd()+'/anatomy/connectivity_998.zip') # connectivity provided by TVB
    conn.configure()

    A = numpy.ones((conn.number_of_regions,1)) * 3.25
    B = numpy.ones((conn.number_of_regions,1)) * 22.0
    a = numpy.ones((conn.number_of_regions,1)) * 0.1
    b = numpy.ones((conn.number_of_regions,1)) * 0.05


    for i in range(3):
        for j in range(3):
            B_value = 19.1 + i * 1.5
            b_value = 0.05 + j * 0.025

            a_value = 0.25
            A_value = 9.5
            
            # jrm = models.JansenRit(mu=numpy.array([0.078]), v0=numpy.array([6.]),p_max=numpy.array([0.15]), p_min=numpy.array([0.03]))
            jrm = models.JansenRit(A=A,B=B,a=a,b=b,mu=numpy.array([0.087]), v0=numpy.array([6.]),p_max=numpy.array([0.15]), p_min=numpy.array([0.03]))

            phi_n_scaling = (0.1 * 3.25 * (jrm.p_max-jrm.p_min) * 0.5 * 2 )**2/2.


            sigma         = numpy.zeros(6) 
            sigma[[3,4]]      = phi_n_scaling



            print("check")
            # the other aspects of the simulator are standard
            sim = simulator.Simulator(
            model=jrm,
            connectivity=conn,
            coupling=coupling.SigmoidalJansenRit(a=numpy.array([1.0])),
            integrator=integrators.HeunStochastic(dt=2 ** -1, noise=noise.Additive(nsig=sigma)),
            monitors=(monitors.Raw(),),
            ).configure()
            print("check")

            out = sim.run(simulation_length=200)

            for iii in range(20):

                hfo_duration = 100
                siml = random.randint(1,899)
                (time1,data1), = sim.run(simulation_length=siml)
                a[region_id] = a_value
                b[region_id] = b_value
                A[region_id] = A_value
                B[region_id] = B_value
                sim.model.a = a.squeeze()
                sim.model.b = b.squeeze()
                sim.model.A = A.squeeze()
                sim.model.B = B.squeeze()
                (time2,data2), = sim.run(simulation_length=hfo_duration)
                a[region_id] = 0.1
                b[region_id] = 0.05
                A[region_id] = 3.25
                B[region_id] = 22.0
                sim.model.a = a.squeeze()
                sim.model.b = b.squeeze()
                sim.model.A = A.squeeze()
                sim.model.B = B.squeeze()
                (time3,data3), = sim.run(simulation_length=1000-hfo_duration-siml)

                time = numpy.concatenate((time1,time2,time3))
                data_all= numpy.concatenate((data1,data2,data3),axis=0)

                data = (data_all[:, 1, :, :] - data_all[:, 2, :, :]).squeeze().astype(numpy.float32)

                # # in the fsaverage5 mapping, there is no vertices corresponding to region 7,325,921, 949, so change label 994-998 to those id
                # data[:, 7] = data[:, 994]
                # data[:, 325] = data[:, 997]
                # data[:, 921] = data[:, 996]
                # data[:, 949] = data[:, 995]
                # data = data[:, :994]

                savemat('hfo_source/raw_nmm_10012024/region{}/B_iter_{}_b_iter_{}_{}.mat'.format(region_id,B_value, b_value, iii),
                {'time': time, 'data': data, 'B': B_value, 'b': b_value,'siml':siml})
                    
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TVB Simulation')
    parser.add_argument('--a_start', type=int, default=0, metavar='t/f', help='start region id')
    parser.add_argument('--a_end', type=int, default=1, metavar='t/f', help='end region id')
    args = parser.parse_args()
    os.environ["MKL_NUM_THREADS"] = "1"
    start_time = time.time()
    # RUN THE CODE IN PARALLEL
    # processes = [mp.Process(target=main, args=(x,)) for x in range(args.a_start, args.a_end)]
    # for p in processes:
    #     p.start()
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    # NO PARALLEL
    for x in range(args.a_start, args.a_end):
        main(x)
    print('Total_time', time.time() - start_time)
