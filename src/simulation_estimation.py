import numpy as np
from tqdm import tqdm

from typing import Callable, Any, List

import pandas as pd

import matplotlib.pyplot as plt

import copy

from abc import ABC, abstractmethod



class SimulationBase(ABC):
    """
    Base class of the simulation framework. 
    It creates a pipeline to test the performance of some estimation methods on simulated data.

    Parameters
    ----------
    n_MonteCarlo : int
        number of Monte Carlo to perform
    estimation_methods : dict of functions
        Estimation methods to be tested. Keys are used as names for
        the methods. Every method should only take the data as input
        argument and only return the estimate of the true parameter.
    error_measures : dict of functions
        Error measures to evaluate the performance. Keys are used as names
        for the error measures. Every error measure should only take the
        parameter and estimate as input argument and return a float.
    index : List or None
        List of values of the parameter that varies in the data
        (e.g., number of samples, snr, etc.).
    index_name : str
        Name of the index.
    """
    def __init__(self,n_MonteCarlo : int, estimation_methods : dict, error_measures : dict,\
                        index=None, index_name="") -> None:
        # store parameters
        self.n_MonteCarlo = n_MonteCarlo
        self.estimation_methods = estimation_methods
        self.error_measures = error_measures
        self.index = index
        self.index_name = index_name
        # init some stuffs
        self.estimates = dict.fromkeys(self.estimation_methods)
        for method in self.estimation_methods:
            self.estimates[method] = [[] for _ in range(self.n_MonteCarlo)]
        #
        self.errors = dict.fromkeys(self.error_measures)
        self.errors_statistics = dict.fromkeys(self.error_measures)
        for error_measure in self.error_measures:
            self.errors[error_measure] = dict.fromkeys(self.estimation_methods)
            self.errors_statistics[error_measure] = dict.fromkeys(self.estimation_methods)
            for method in self.estimation_methods:
                self.errors[error_measure][method] = [[] for _ in range(self.n_MonteCarlo)]
                self.errors_statistics[error_measure][method] = dict(medians=None, quantiles_95=None, quantiles_05=None, means=None, stds=None)

    @abstractmethod
    def run(self) -> None:
        """
        Performs estimation with every method on simulated data for all Monte Carlo runs.
        Also computes errors between estimates and true parameter for every error measure.
        """

    def compute_statistics(self) -> None:
        """
        Computes the statistics of the errors along all Monte Carlo runs.
        """
        # there should be some checks that run was done
        # add some external modularity on statistics that are computed ?
        for error_measure in self.error_measures:
            for method in self.estimation_methods:
                self.errors_statistics[error_measure][method]['medians'] = np.median(self.errors[error_measure][method], axis=0)
                self.errors_statistics[error_measure][method]['quantiles_95'] = np.quantile(self.errors[error_measure][method], q=0.95, axis=0)
                self.errors_statistics[error_measure][method]['quantiles_05'] = np.quantile(self.errors[error_measure][method], q=0.05, axis=0)
                self.errors_statistics[error_measure][method]['means'] = np.mean(self.errors[error_measure][method], axis=0)
                self.errors_statistics[error_measure][method]['stds'] = np.std(self.errors[error_measure][method], axis=0)
    
    def preview_figures(self, xscale: str = "linear", show_all_runs: bool = False):
        """
        Visualize the performance (in dB) of the estimation methods as a function of the number of samples (log scale).
        Top: medians (lines) and quantiles (colored areas).
        Bottom: means
        """
        for error_measure in self.error_measures:
            plt.figure()
            plt.suptitle(error_measure)
            plt.subplot(211)
            ax = plt.gca()
            for method in self.estimation_methods:
                color = next(ax._get_lines.prop_cycler)['color']
                if self.index is not None:
                    plt.plot(self.index,10*np.log10(self.errors_statistics[error_measure][method]['medians']),label=method,color=color)
                    plt.fill_between(self.index,10*np.log10(self.errors_statistics[error_measure][method]['quantiles_05']), \
                                     10*np.log10(self.errors_statistics[error_measure][method]['quantiles_95']),alpha=0.2,label='_nolegend_',color=color)
                else:
                    index = len(self.errors_statistics[error_measure][method]['medians'])
                    plt.plot(range(index),10*np.log10(self.errors_statistics[error_measure][method]['medians']),label=method,color=color)
                    plt.fill_between(range(index),10*np.log10(self.errors_statistics[error_measure][method]['quantiles_05']), \
                                     10*np.log10(self.errors_statistics[error_measure][method]['quantiles_95']),alpha=0.2,label='_nolegend_',color=color)
            plt.legend(loc='best')
            plt.title('Medians, 5% and 95% quantiles')
            plt.xlabel(self.index_name)
            plt.ylabel('error (dB)')
            plt.xscale(xscale)
            #
            plt.subplot(212)
            ax = plt.gca()
            for method in self.estimation_methods:
                color = next(ax._get_lines.prop_cycler)['color']
                #
                if show_all_runs is True:
                    for monte_carlo in range(self.n_MonteCarlo):
                        ix = len(self.errors[error_measure][method][monte_carlo])
                        plt.plot(range(ix),10*np.log10(self.errors[error_measure][method][monte_carlo]),alpha=0.1,label='_nolegend_',color=color)
                #
                if self.index is not None:
                    plt.plot(self.index,10*np.log10(self.errors_statistics[error_measure][method]['means']),label=method,color=color)
                else:
                    index = len(self.errors_statistics[error_measure][method]['means'])
                    plt.plot(range(index),10*np.log10(self.errors_statistics[error_measure][method]['means']),label=method,color=color)
            plt.legend(loc='best')
            plt.title('Means')
            plt.xlabel(self.index_name)
            plt.ylabel('error (dB)')
            plt.xscale(xscale)
            #
            plt.tight_layout()
        plt.show()

    def export_statistics_to_csv(self, folder: str = "./", write_index: bool = True):
        """
        Export the errors statistics to a csv file.

        Parameters
        ----------
        folder : str, optional
            folder the csv file should be written in, by default "./"
        write_index : bool, optional
            whether to write the index in the csv files
        """
        for error_measure in self.error_measures:
            filename = folder + "perf_vs_" + self.index_name + "_" + error_measure + ".csv"
            df = pd.concat([pd.DataFrame(self.errors_statistics[error_measure][method]).add_prefix(method+"_") for method in self.estimation_methods],axis=1)
            if write_index:
                df.index = self.index
                df.index.name = self.index_name
            df.to_csv(filename,index=write_index)

    def export_errors_to_csv(self, folder = './', write_index: bool = True):
        """
        Export the errors to csv files.
        One csv file for each method and each error measure.

        Parameters
        ----------
        folder : str, optional
            folder the csv files should be written in, by default "./"
        write_index : bool, optional
            whether to write the index in the csv files
        """
        for error_measure in self.error_measures:
            for method in self.estimation_methods:
                filename = folder + "errors_vs_" + self.index_name + "_" + method + "_" + error_measure + ".csv"
                df = pd.DataFrame(self.errors[error_measure][method]).T.add_prefix('run_')
                if write_index:
                    df.index = self.index
                    df.index.name = self.index_name
                df.to_csv(filename,index=write_index)



class SimulationPerfvsSamples(SimulationBase):
    """
    Simulation framework. It creates a pipeline to test the performance of
    some estimation methods on simulated data for various numbers of samples.

    Parameters
    ----------
    random_parameter : function
        function that draws a random parameter to be estimated.
        It should not have any argument. 
    random_data : function
        function that draws random data from a given parameter.
        Its arguments are the parameter, the number of samples
        and the number of Monte Carlo to generate. It should return 
        a ndarray (or list) of at least 3 dimensions, the two first corresponding
        to n_MonteCarlo and max(n_samples_all), respectively.
    n_samples_all : list of int
        list of all the numbers of samples to consider.
        Probably best if sorted in ascending order.
    n_MonteCarlo : int
        number of Monte Carlo to perform
    estimation_methods : dict of functions
        Estimation methods to be tested. Keys are used as names for
        the methods. Every method should only take the data as input
        argument and only return the estimate of the true parameter.
    error_measures : dict of functions
        Error measures to evaluate the performance. Keys are used as names
        for the error measures. Every error measure should only take the
        parameter and estimate as input argument and return a float. 

    Attributes
    ----------
    parameter : ndarray
        random parameter generated with random_parameter().
    data : ndarray (or list)
        random data generated with random_data(parameter, max(n_samples_all), n_MonteCarlo).
    estimates : dict of lists
        estimates of the true parameter for every method, number of samples and Monte Carlo run.
    errors : dict of dict of lists
        errors between estimates and the true parameter for every error measure, method, number of samples and Monte Carlo run.
    errors_statistics : dict of dict of dict of lists
        for every error measure, method and number of samples, statistics (medians, 5% and 95% quantiles, means and stds)
        of the errors along Monte Carlo runs.
    """
    def __init__(self, random_parameter: Callable[[],Any], random_data: Callable[[Any,int,int],list], n_samples_all: list, \
                 n_MonteCarlo: int, estimation_methods: dict, error_measures: dict) -> None:
        # store some input variables
        self.random_parameter = random_parameter
        self.random_data = random_data
        self.n_samples_all = n_samples_all
        # compute parameter and data
        self.parameter = self.random_parameter()
        self.data = self.random_data(self.parameter,np.max(n_samples_all),n_MonteCarlo)
        # initialize base
        super().__init__(n_MonteCarlo=n_MonteCarlo, estimation_methods=estimation_methods, error_measures=error_measures,\
                        index=n_samples_all, index_name="nsamples") 

    def run(self) -> None:
        # we need to parallelize that as much as possible, but proof of concept now
        for it in tqdm(range(self.n_MonteCarlo)):
            for n_samples in self.n_samples_all:
                for method in self.estimation_methods:
                    estimate = self.estimation_methods[method](self.data[it][:n_samples])
                    self.estimates[method][it].append(estimate)
                    for error_measure in self.error_measures:
                        self.errors[error_measure][method][it].append(self.error_measures[error_measure](self.parameter,estimate))

    def preview_figures(self):
        return super().preview_figures(xscale="log", show_all_runs=False)
    
    def export_statistics_to_csv(self, folder: str = "./"):
        return super().export_statistics_to_csv(folder, write_index=True)
    
    def export_errors_to_csv(self, folder='./'):
        return super().export_errors_to_csv(folder, write_index=True)



class SimulationPerfvsIterations(SimulationBase):
    """
    Simulation framework. It creates a pipeline to look at the performance of
    some iterative estimation methods on simulated data across iterations.

    Parameters
    ----------
    random_parameter : function
        function that draws a random parameter to be estimated.
        It should not have any argument. 
    random_data : function
        function that draws random data from a given parameter.
        Its arguments are the parameter, the number of samples
        and the number of Monte Carlo to generate. It should return 
        a ndarray (or list) of at least 3 dimensions, the two first corresponding
        to n_MonteCarlo and n_samples, respectively.
    n_samples : int
        number of samples for simulated data.
    n_MonteCarlo : int
        number of Monte Carlo to perform
    estimation_methods : dict of functions
        Estimation methods to be tested. Keys are used as names for
        the methods. Every method should only take the data as input
        argument and only return the list of iterates.
    error_measures : dict of functions
        Error measures to evaluate the performance. Keys are used as names
        for the error measures. Every error measure should only take the
        parameter and estimate as input argument and return a float. 

    Attributes
    ----------
    parameter : ndarray
        random parameter generated with random_parameter().
    data : ndarray (or list)
        random data generated with random_data(parameter, n_samples, n_MonteCarlo).
    iterates : dict of lists
        iterates of every method for each Monte Carlo run.
    errors : dict of dict of lists
        errors between every iterations of all methods and the true parameter for each error measure and Monte Carlo run.
    errors_statistics : dict of dict of dict of lists
        for every error measure, method and maximum number of iterations, statistics (medians, 5% and 95% quantiles, means and stds)
        of the errors along Monte Carlo runs.
    """
    def __init__(self, random_parameter: Callable[[],Any], random_data: Callable[[Any,int,int],list], n_samples: int, \
                  n_MonteCarlo: int, estimation_methods: dict, error_measures: dict) -> None:
        # store some input variables
        self.random_parameter = random_parameter
        self.random_data = random_data
        self.n_samples = n_samples
        # compute parameter and data
        self.parameter = self.random_parameter()
        self.data = self.random_data(self.parameter,n_samples,n_MonteCarlo)
        # initialize base
        super().__init__(n_MonteCarlo=n_MonteCarlo, estimation_methods=estimation_methods, error_measures=error_measures, index=None, index_name="iterations")

    def run(self) -> None:
        # we need to parallelize that as much as possible, but proof of concept now
        for it in tqdm(range(self.n_MonteCarlo)):
            for method in self.estimation_methods:
                iterates = self.estimation_methods[method](self.data[it])
                self.estimates[method][it].append(iterates)
                for error_measure in self.error_measures:
                    for iterate in iterates:
                        self.errors[error_measure][method][it].append(self.error_measures[error_measure](self.parameter,iterate))

    def compute_statistics(self) -> None:
        for error_measure in self.error_measures:
            for method in self.estimation_methods:
                # complete data
                err_tmp = copy.deepcopy(self.errors[error_measure][method]) # we don't want to touch self.errors
                lens = [len(err) for err in err_tmp]
                for err in err_tmp:
                    while len(err)<max(lens):
                        err.append(err[-1])
                # compute the stats
                self.errors_statistics[error_measure][method]['medians'] = np.median(err_tmp, axis=0)
                self.errors_statistics[error_measure][method]['quantiles_95'] = np.quantile(err_tmp, q=0.95, axis=0)
                self.errors_statistics[error_measure][method]['quantiles_05'] = np.quantile(err_tmp, q=0.05, axis=0)
                self.errors_statistics[error_measure][method]['means'] = np.mean(err_tmp, axis=0)

    def preview_figures(self):
        return super().preview_figures(xscale="linear", show_all_runs=True)
    
    def export_statistics_to_csv(self, folder: str = "./"):
        return super().export_statistics_to_csv(folder, write_index=False)
    
    def export_errors_to_csv(self, folder='./'):
        return super().export_errors_to_csv(folder, write_index=False)


# class SimulationPerfvsSamples_old():
#     """
#     Simulation framework. It creates a pipeline to test the performance of
#     some estimation methods on simulated data for various numbers of samples.

#     Parameters
#     ----------
#     random_parameter : function
#         function that draws a random parameter to be estimated.
#         It should not have any argument. 
#     random_data : function
#         function that draws random data from a given parameter.
#         Its arguments are the parameter, the number of samples
#         and the number of Monte Carlo to generate. It should return 
#         a ndarray (or list) of at least 3 dimensions, the two first corresponding
#         to n_MonteCarlo and max(n_samples_all), respectively.
#     n_samples_all : list of int
#         list of all the numbers of samples to consider.
#         Probably best if sorted in ascending order.
#     n_MonteCarlo : int
#         number of Monte Carlo to perform
#     estimation_methods : dict of functions
#         Estimation methods to be tested. Keys are used as names for
#         the methods. Every method should only take the data as input
#         argument and only return the estimate of the true parameter.
#     error_measures : dict of functions
#         Error measures to evaluate the performance. Keys are used as names
#         for the error measures. Every error measure should only take the
#         parameter and estimate as input argument and return a float. 

#     Attributes
#     ----------
#     parameter : ndarray
#         random parameter generated with random_parameter().
#     data : ndarray (or list)
#         random data generated with random_data(parameter, max(n_samples_all), n_MonteCarlo).
#     estimates : dict of lists
#         estimates of the true parameter for every method, number of samples and Monte Carlo run.
#     errors : dict of dict of lists
#         errors between estimates and the true parameter for every error measure, method, number of samples and Monte Carlo run.
#     errors_statistics : dict of dict of dict of lists
#         for every error measure, method and number of samples, statistics (medians, 5% and 95% quantiles, means and stds)
#         of the errors along Monte Carlo runs.
#     """
#     def __init__(self, random_parameter: Callable[[],Any], random_data: Callable[[Any,int,int],list], n_samples_all: list, n_MonteCarlo: int, estimation_methods: dict, error_measures: dict) -> None:
#         # store input variables
#         self.random_parameter = random_parameter
#         self.random_data = random_data
#         self.n_samples_all = n_samples_all
#         self.n_MonteCarlo = n_MonteCarlo
#         self.estimation_methods = estimation_methods # dictionary (this way we have names we can exploit)
#         self.error_measures = error_measures # dictionary
#         # compute parameter and data
#         self.parameter = self.random_parameter()
#         self.data = self.random_data(self.parameter,np.max(self.n_samples_all),self.n_MonteCarlo)
        # # initialize some stuff
        # self.estimates = dict.fromkeys(self.estimation_methods)
        # for method in self.estimation_methods:
        #     self.estimates[method] = [[] for _ in range(self.n_MonteCarlo)]
        # #
        # self.errors = dict.fromkeys(self.error_measures)
        # self.errors_statistics = dict.fromkeys(self.error_measures)
        # for error_measure in self.error_measures:
        #     self.errors[error_measure] = dict.fromkeys(self.estimation_methods)
        #     self.errors_statistics[error_measure] = dict.fromkeys(self.estimation_methods)
        #     for method in self.estimation_methods:
        #         self.errors[error_measure][method] = [[] for _ in range(self.n_MonteCarlo)]
        #         self.errors_statistics[error_measure][method] = dict(medians=None, quantiles_95=None, quantiles_05=None, means=None, stds=None)

#     def run(self):
#         """
#         Performs estimation with every method on simulated data for all numbers of samples and Monte Carlo runs.
#         Also computes errors between estimates and true parameter for every error measure.
#         """
#         # we need to parallelize that as much as possible, but proof of concept now
#         for it in tqdm(range(self.n_MonteCarlo)):
#             for n_samples in self.n_samples_all:
#                 for method in self.estimation_methods:
#                     estimate = self.estimation_methods[method](self.data[it][:n_samples])
#                     self.estimates[method][it].append(estimate)
#                     for error_measure in self.error_measures:
#                         self.errors[error_measure][method][it].append(self.error_measures[error_measure](self.parameter,estimate))

#     def compute_statistics(self):
#         """
#         Computes the statistics of the errors along all Monte Carlo runs.
#         """
#         # there should be some checks that run was done
#         for error_measure in self.error_measures:
#             for method in self.estimation_methods:
#                 self.errors_statistics[error_measure][method]['medians'] = np.median(self.errors[error_measure][method], axis=0)
#                 self.errors_statistics[error_measure][method]['quantiles_95'] = np.quantile(self.errors[error_measure][method], q=0.95, axis=0)
#                 self.errors_statistics[error_measure][method]['quantiles_05'] = np.quantile(self.errors[error_measure][method], q=0.05, axis=0)
#                 self.errors_statistics[error_measure][method]['means'] = np.mean(self.errors[error_measure][method], axis=0)
#                 self.errors_statistics[error_measure][method]['stds'] = np.std(self.errors[error_measure][method], axis=0)

#     def preview_figures(self):
#         """
#         Visualize the performance (in dB) of the estimation methods as a function of the number of samples (log scale).
#         Top: medians (lines) and quantiles (colored areas).
#         Bottom: means
#         """
#         for error_measure in self.error_measures:
#             plt.figure()
#             plt.suptitle(error_measure)
#             plt.subplot(211)
#             for method in self.estimation_methods:
#                 plt.plot(self.n_samples_all,10*np.log10(self.errors_statistics[error_measure][method]['medians']),label=method)
#                 plt.fill_between(self.n_samples_all,10*np.log10(self.errors_statistics[error_measure][method]['quantiles_05']), 10*np.log10(self.errors_statistics[error_measure][method]['quantiles_95']),alpha=0.2,label='_nolegend_')
#             plt.legend(loc='best')
#             plt.title('Medians, 5% and 95% quantiles')
#             plt.xlabel('number of samples')
#             plt.ylabel('error (dB)')
#             plt.xscale('log')
#             #
#             plt.subplot(212)
#             for method in self.estimation_methods:
#                 plt.plot(self.n_samples_all,10*np.log10(self.errors_statistics[error_measure][method]['means']),label=method)
#             plt.legend(loc='best')
#             plt.title('Means')
#             plt.xlabel('number of samples')
#             plt.ylabel('error (dB)')
#             plt.xscale('log')
#             #
#             plt.tight_layout()
#         plt.show()
    
#     def export_statistics_to_csv(self,folder: str ="./"):
#         """
#         Export the errors statistics to a csv file.

#         Parameters
#         ----------
#         folder : str, optional
#             folder the csv file should be written in, by default "./"
#         """
#         for error_measure in self.error_measures:
#             filename = folder + "perf_vs_samples_" + error_measure + ".csv"
#             df = pd.concat([pd.DataFrame(self.errors_statistics[error_measure][method]).add_prefix(method+"_") for method in self.estimation_methods],axis=1)
#             df.index = self.n_samples_all
#             df.index.name = "n_samples"
#             df.to_csv(filename)

#     def export_errors_to_csv(self,folder='./'):
#         """
#         Export the errors to csv files.
#         One csv file for each method and each error measure.

#         Parameters
#         ----------
#         folder : str, optional
#             folder the csv files should be written in, by default "./"
#         """
#         for error_measure in self.error_measures:
#             for method in self.estimation_methods:
#                 filename = folder + "errors_vs_samples_" + method + "_" + error_measure + ".csv"
#                 df = pd.DataFrame(self.errors[error_measure][method]).T.add_prefix('run_')
#                 df.index = self.n_samples_all
#                 df.index.name = "n_samples"
#                 df.to_csv(filename)




# class SimulationPerfvsIterations_old():
#     """
#     Simulation framework. It creates a pipeline to look at the performance of
#     some iterative estimation methods on simulated data across iterations.

#     Parameters
#     ----------
#     random_parameter : function
#         function that draws a random parameter to be estimated.
#         It should not have any argument. 
#     random_data : function
#         function that draws random data from a given parameter.
#         Its arguments are the parameter, the number of samples
#         and the number of Monte Carlo to generate. It should return 
#         a ndarray (or list) of at least 3 dimensions, the two first corresponding
#         to n_MonteCarlo and n_samples, respectively.
#     n_samples : int
#         number of samples for simulated data.
#     n_MonteCarlo : int
#         number of Monte Carlo to perform
#     estimation_methods : dict of functions
#         Estimation methods to be tested. Keys are used as names for
#         the methods. Every method should only take the data as input
#         argument and only return the list of iterates.
#     error_measures : dict of functions
#         Error measures to evaluate the performance. Keys are used as names
#         for the error measures. Every error measure should only take the
#         parameter and estimate as input argument and return a float. 

#     Attributes
#     ----------
#     parameter : ndarray
#         random parameter generated with random_parameter().
#     data : ndarray (or list)
#         random data generated with random_data(parameter, n_samples, n_MonteCarlo).
#     iterates : dict of lists
#         iterates of every method for each Monte Carlo run.
#     errors : dict of dict of lists
#         errors between every iterations of all methods and the true parameter for each error measure and Monte Carlo run.
#     errors_statistics : dict of dict of dict of lists
#         for every error measure, method and maximum number of iterations, statistics (medians, 5% and 95% quantiles, means and stds)
#         of the errors along Monte Carlo runs.
#     """
#     def __init__(self, random_parameter, random_data, n_samples: int, n_MonteCarlo: int, estimation_methods: dict, error_measures: dict):
#         # store input variables
#         self.random_parameter = random_parameter
#         self.random_data = random_data
#         self.n_samples = n_samples
#         self.n_MonteCarlo = n_MonteCarlo
#         self.estimation_methods = estimation_methods
#         self.error_measures = error_measures
#         # compute parameter and data
#         self.parameter = self.random_parameter()
#         self.data = self.random_data(self.parameter,self.n_samples,self.n_MonteCarlo)
#         # initialize some stuff
#         self.iterates = dict.fromkeys(self.estimation_methods)
#         for method in self.estimation_methods:
#             self.iterates[method] = [[] for _ in range(self.n_MonteCarlo)]
#         self.errors = dict.fromkeys(self.error_measures)
#         self.errors_statistics = dict.fromkeys(self.error_measures)
#         for error_measure in self.error_measures:
#             self.errors[error_measure] = dict.fromkeys(self.estimation_methods)
#             self.errors_statistics[error_measure] = dict.fromkeys(self.estimation_methods)
#             for method in self.estimation_methods:
#                 self.errors[error_measure][method] = [[] for _ in range(self.n_MonteCarlo)]
#                 self.errors_statistics[error_measure][method] = dict(medians=[], quantiles_95=[], quantiles_05=[], means=[])

#     def run(self):
#         """
#         Performs estimation with every method on simulated data for all Monte Carlo runs.
#         Also computes errors between every iterates and the true parameter for each error measure.
#         """
#         # we need to parallelize that as much as possible, but proof of concept now
#         for it in tqdm(range(self.n_MonteCarlo)):
#             for method in self.estimation_methods:
#                 iterates = self.estimation_methods[method](self.data[it])
#                 self.iterates[method][it].append(iterates)
#                 for error_measure in self.error_measures:
#                     for iterate in iterates:
#                         self.errors[error_measure][method][it].append(self.error_measures[error_measure](self.parameter,iterate))

#     def compute_statistics(self):
#         """
#         Computes the statistics of the errors along all Monte Carlo runs.
#         """
#         for error_measure in self.error_measures:
#             for method in self.estimation_methods:
#                 # complete data
#                 err_tmp = copy.deepcopy(self.errors[error_measure][method]) # we don't want to touch self.errors
#                 lens = [len(err) for err in err_tmp]
#                 for err in err_tmp:
#                     while len(err)<max(lens):
#                         err.append(err[-1])
#                 # compute the stats
#                 self.errors_statistics[error_measure][method]['medians'] = np.median(err_tmp, axis=0)
#                 self.errors_statistics[error_measure][method]['quantiles_95'] = np.quantile(err_tmp, q=0.95, axis=0)
#                 self.errors_statistics[error_measure][method]['quantiles_05'] = np.quantile(err_tmp, q=0.05, axis=0)
#                 self.errors_statistics[error_measure][method]['means'] = np.mean(err_tmp, axis=0)

#     def preview_figures(self):
#         """
#         Visualize the performance (in dB) of the estimation methods as a function of the number of samples.
#         Top: medians (lines) and quantiles (colored areas).
#         Bottom: means and every single run (blurred lines)
#         """
#         for error_measure in self.error_measures:
#             plt.figure()
#             plt.suptitle(error_measure)
#             plt.subplot(211)
#             ax = plt.gca()
#             for method in self.estimation_methods:
#                 color = next(ax._get_lines.prop_cycler)['color']
#                 max_it = len(self.errors_statistics[error_measure][method]['medians'])
#                 plt.plot(range(max_it),10*np.log10(self.errors_statistics[error_measure][method]['medians']),label=method,color=color)
#                 plt.fill_between(range(max_it),10*np.log10(self.errors_statistics[error_measure][method]['quantiles_05']), 10*np.log10(self.errors_statistics[error_measure][method]['quantiles_95']),alpha=0.2,label='_nolegend_',color=color)
#             plt.legend(loc='best')
#             plt.title('Medians, 5% and 95% quantiles')
#             plt.xlabel('iterations')
#             plt.ylabel('error (dB)')
#             # plt.xscale('log')
#             #
#             plt.subplot(212)
#             ax = plt.gca()
#             for method in self.estimation_methods:
#                 color = next(ax._get_lines.prop_cycler)['color']
#                 for monte_carlo in range(self.n_MonteCarlo):
#                     n_it = len(self.errors[error_measure][method][monte_carlo])
#                     plt.plot(range(n_it),10*np.log10(self.errors[error_measure][method][monte_carlo]),alpha=0.1,label='_nolegend_',color=color)
#                 max_it = len(self.errors_statistics[error_measure][method]['means'])
#                 plt.plot(range(max_it),10*np.log10(self.errors_statistics[error_measure][method]['means']),label=method,color=color)
#             plt.legend(loc='best')
#             plt.title('Means')
#             plt.xlabel('iterations')
#             plt.ylabel('error (dB)')
#             # plt.xscale('log')
#             #
#             plt.tight_layout()
#         plt.show()

#     def export_statistics_to_csv(self,folder="./"):
#         """
#         Export the errors statistics to a csv file.

#         Parameters
#         ----------
#         folder : str, optional
#             folder the csv file should be written in, by default "./"
#         """
#         for error_measure in self.error_measures:
#             filename = folder + "perf_vs_iterations_" + error_measure + ".csv"
#             df = pd.concat([pd.DataFrame(self.errors_statistics[error_measure][method]).add_prefix(method+"_") for method in self.estimation_methods],axis=1)
#             df.to_csv(filename,index=False)

#     def export_errors_to_csv(self,folder='./'):
#         """
#         Export the errors to csv files.
#         One csv file for each method and each error measure.

#         Parameters
#         ----------
#         folder : str, optional
#             folder the csv files should be written in, by default "./"
#         """
#         for error_measure in self.error_measures:
#             for method in self.estimation_methods:
#                 filename = folder + "errors_vs_iterations_" + method + "_" + error_measure + ".csv"
#                 df = pd.DataFrame(self.errors[error_measure][method]).T.add_prefix('run_')
#                 df.to_csv(filename,index=False)

