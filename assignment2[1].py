#################################
# Your name: Roei Ben Zion
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math
from sklearn.model_selection import train_test_split

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = sorted(np.random.rand(m))
        samp = np.array([0,1])
        ys = [0]*m
        for i in range(len(xs)):
            x = xs[i]
            if(self.in_good_interval(x)):
                rnd = np.random.choice(samp, p=[0.2, 0.8])
            else:
                rnd = np.random.choice(samp, p=[0.9, 0.1])
            ys[i] = rnd
        lst = list(zip(xs, ys))
        ret = np.array(lst)
        return ret
        # TODO: Implement me


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        count = int(((m_last-m_first)/step))+1
        true_vector = [0.0]*count
        empirical_vector = [0]*count
        for t in range(1, 100):
            i = 0
            for n in range(m_first, m_last+1, step):
                S = self.sample_from_D(n)
                xs = S[:, 0]
                ys = S[:, 1]
                curr_intervals,empirical_error = intervals.find_best_interval(xs=xs, ys=ys, k=3)
                empirical_error  = empirical_error/n
                true_error = self.calc_true_error(curr_intervals)
                true_vector[i] += true_error/T
                empirical_vector[i] += empirical_error/T
                i += 1                
        plt.plot(np.arange(m_first, m_last+1, step), true_vector)
        plt.plot(np.arange(m_first, m_last+1, step), empirical_vector)
        plt.show()
        lst = list(zip(true_vector, empirical_vector))
        ret = np.array(lst)
        return ret
        # TODO: Implement the loop
    
    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        S = self.sample_from_D(m)
        xs = S[:, 0]
        ys = S[:, 1]
        min_k_val = float("inf")
        min_k = float("inf")
        count = int(((k_last-k_first)/step))+1
        true_vector = [0]*count
        empirical_vector = [0]*count
        i = 0
        for k in range(k_first, k_last+1, step):
            curr_intervals,empirical_error = intervals.find_best_interval(xs=xs, ys=ys, k=k)
            empirical_error = empirical_error/m
            if(min_k_val > empirical_error):
                min_k_val = empirical_error
                min_k = k
            empirical_vector[i] = empirical_error
            true_vector[i] = self.calc_true_error(curr_intervals)
            i+=1
        plt.plot(np.arange(1, count+1) ,true_vector)
        plt.plot(np.arange(1, count+1),empirical_vector)
        plt.show()
        return min_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        S = self.sample_from_D(m)
        xs = S[:, 0]
        ys = S[:, 1]
        min_k_erm_val = float("inf")
        min_k_srm_val = float("inf")
        min_k_srm = float("inf")

        count = int(((k_last-k_first)/step))+1
        true_vector = [0]*(count)
        empirical_vector = [0]*(count)
        penalty_vector = [0]*count
        sum_penalty_empirical_vector = [0]*count
        plt_vector = np.arange(1, count+1)

        i = 0
        for k in range(k_first, k_last+1, step):
            penalty = 2*math.sqrt(((2*k) + math.log(20, math.e))/m)
            curr_intervals,empirical_error = intervals.find_best_interval(xs=xs, ys=ys, k=k)

            if(min_k_erm_val > empirical_error):
                min_k_erm_val = empirical_error
            
            if(min_k_srm_val > empirical_error + penalty):
                min_k_srm_val = empirical_error + penalty
                min_k_srm = k
            
            empirical_error = empirical_error/m
            empirical_vector[i] = empirical_error
            true_vector[i] = self.calc_true_error(curr_intervals)
            penalty_vector[i] = penalty
            sum_penalty_empirical_vector[i] = empirical_error+penalty
            i += 1
        
        plt.plot(plt_vector, true_vector, label = 'True error')
        plt.plot(plt_vector, empirical_vector, label = 'Empirical_error')
        plt.plot(plt_vector, penalty_vector, label = 'penalty')
        plt.plot(plt_vector, sum_penalty_empirical_vector, label = 'SRM')
        plt.legend()
        plt.show()
        return min_k_srm
    
    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        S = self.sample_from_D(m)
        xs = S[:, 0]
        ys = S[:, 1]
        xs_train, xs_validation, ys_train, ys_validation = train_test_split(xs, ys, train_size=0.8)
        lst = np.array([xs_train, ys_train]).T
        lst = lst[xs_train.argsort()]
        xs_train = lst[:,0]
        ys_train = lst[:,1]

        lst = np.array([xs_validation, ys_validation]).T
        lst = lst[xs_validation.argsort()]
        xs_validation = lst[:,0]
        ys_validation = lst[:,1]

        erm_vector = [0]*10
        min_emp = float("inf")
        min_k = float("inf")
        for k in range(1, 11):
            curr_intervals,empirical_error = intervals.find_best_interval(xs=xs_train, ys=ys_train, k=k)
            erm_vector[k-1] = curr_intervals
        k = 1
        for h in erm_vector:
            empirical_error = self.calc_empirical_error(h, xs_validation, ys_validation)
            if(empirical_error < min_emp):
                min_emp = empirical_error
                min_k = k
            k+=1
        # TODO: Implement me
        print("The best K is ", min_k)
        print("The best hypotheses is ", erm_vector[min_k-1])
        return erm_vector[min_k]

    #################################
    def calc_empirical_error(self, I:list, x:list, y:list) -> float:
        count = 0
        n = len(x)
        for i in range(n):
            instance = x[i]
            label = y[i]
            b = False
            for interval in I:
                if(instance >= interval[0] and instance <= interval[1]):
                    b = True
                    break
            if (not b and label == 1.0):
                count += 1
            if (b and label == 0.0):
                count += 1
        return count/(n)
    def in_good_interval(self, a:float) -> bool:
        return (0<=a and a<=0.2) or (0.4<=a and a<=0.6) or (0.8<=a and a<=1)

    
    #P(X in I , X in good)
    def find_good_prob_I(self, a:float, b:float) -> float:
        x = 0
        x += max(0,(min(b, 0.2) - max(a,0)))
        x += max(0,(min(b, 0.6) - max(a,0.4)))
        x += max(0,(min(b, 1) - max(a,0.8)))
        return x

    def calc_true_error(self, I: list):
        '''
        Explanation about the true error calculation in the theoretical submition
        '''
        true_error = 0
        p_good_I = 0
        p_bad_I = 0
        p_good_not_I = 0
        p_bad_not_I = 0
        p_x_not_i = 0

        #X in I
        for interval in I:
            a = interval[0]
            b = interval[1]
            temp_p_good = self.find_good_prob_I(a, b)
            p_good_I += temp_p_good
            p_bad_I += (b-a)-temp_p_good
            p_x_not_i += (b-a)
        
        p_good_not_I = 0.6 - p_good_I
        p_x_not_i = 1 - p_x_not_i
        p_bad_not_I = p_x_not_i - p_good_not_I
        temp = 0.2*p_good_I + 0.9*p_bad_I + 0.1*p_bad_not_I + 0.8*p_good_not_I
        true_error += temp
        return true_error

    #P(X in I , X in good) = P(X in I) - P(X in I, X not good)
    #P(X in I , X in good) = P(X in good) - P(X not in I , X in good)
    #P(X not in I , X in good) = P(X not in I) - P(X not in I , X not in good)
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500) 
