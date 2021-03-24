# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:51:35 2019

@author: Marcezar
"""

import os
import pickle

# Import NumPy
import numpy as np

# Import MathPlotLib
import matplotlib.pyplot as plt
#from matplotlib import rcParams
#rcParams['text.usetex'] = True

from IPython.display import clear_output





def CG_getResult():
    # Create and print a list of available matrices
    matrixlist = [A_name for A_name in os.listdir('results') if A_name[-2:] !=".p"]
    for i, A_name in enumerate(matrixlist):
        print(i,A_name)
        
    # Choose matrix
    choice = input()
    if choice == 'quit':
        print("aborted")
    try:
        A_name = matrixlist[int(choice)]
    except Exception:
        print("wrong input")
    clear_output()
        
    # Create and print a list of available tests
    tests_one = os.listdir('results\\'+A_name)       
    for i, resultname in enumerate(tests_one):
        print(i,resultname)        
        
    # Choose result
    choice = input()
    if choice == 'quit':
        print("aborted")
    try:
        resultname = tests_one[int(choice)]
    except Exception:
        print("wrong input")
    clear_output()
    
    return pickle.load(open('results\\'+A_name+'\\'+resultname,'rb'))





def PlotMatrices():
    matrixlist = [A_name for A_name in os.listdir('results') if A_name[-2:] !=".p"]
    print("Total of",len(matrixlist),"matrices.")
    for i, A_name in enumerate(matrixlist):
        print(i,A_name,pickle.load(open('results\\' + A_name + '.p','rb')).shape)    
        plt.spy(pickle.load(open('results\\' + A_name + '.p','rb')))
        plt.show()





def CG_plot_all(results,A):
    name = results[0]['MatrixName']
    fig = plt.figure(figsize=(18,18))
    fig.subplots_adjust(hspace=0.3,wspace=0.2)
    
    # Plot setup computation time
    plt.subplot(3,2,1)
    CG_plot_part(results,'TimeSetup','Setup computation time')
    
    # Plot CG computation time
    plt.subplot(3,2,2)
    CG_plot_part(results,'TimeCG','CG computation time')
    
    # Plot overall computation time
    plt.subplot(3,2,3)
    CG_plot_part(results,'TimeOverall','Overall computation time')
    
    # Plot matrix structure
    plt.subplot(3,2,4)
    plt.spy(A, color='black', origin='lower')
    plt.xlabel('column')
    plt.ylabel('row')
    plt.title('Density plot of matrix ' + name)
    
    # Plot comparison of relative residuals
    plt.subplot(3,2,5)
    CG_plot_part(results,'Residual','Comparison of residuals')

    # Plot comparison of relative errors
    plt.subplot(3,2,6)
    CG_plot_part(results,'Error','Comparison of errors')
    
    plt.legend(loc='best')
    plt.show()
    
    # Output of name and condition number of the matrix
    print("CG-Analysis of matrix " + name + " complete.")
    print("Condition number computed in 2-norm: %.3e" % np.linalg.cond(A.toarray()))
    #largSingVal = spsl.svds(A,k=1,which='LM',return_singular_vectors=False)
    #smalSingVal = spsl.svds(A,k=1,which='SM',return_singular_vectors=False)
    #print("Condition number:", largSingVal/smalSingVal)





def CG_plot_TMI():
    # Create and print a list of available matrices
    matrixlist = [A_name for A_name in os.listdir('results') if A_name[-2:] !=".p"]
    for i, A_name in enumerate(matrixlist):
        print(i,A_name)
    
    # Choose number of matrices for testing
    usedmatrices = []
    morematrices = True
    while morematrices:
        choice = input()
        if choice == 'quit':
            return "aborted"
        elif choice == 'all':
            usedmatrices = matrixlist
            morematrices = False
            break
        try:
            usedmatrices.append(matrixlist[int(choice)])
        except Exception:
            morematrices = False
    clear_output()
    
    tests_one = os.listdir('results\\'+usedmatrices[0])       
    for i, resultname in enumerate(tests_one):
            print(i,resultname)        
    
    # Choose a number of desired tests
    testnumbers = []    
    moretests = True    
    while moretests:
        choice = input()
        try:
            choice = int(choice)
            if choice in range(len(tests_one)):
                testnumbers.append(choice)
            else:
                moretests = False
        except Exception:
            moretests = False
    clear_output()
    
    # gather informations
    for A_name in usedmatrices:
        tests = [os.listdir('results\\'+A_name)[i] for i in testnumbers]
        
        results = []
        for result in tests:
            results.append(pickle.load(open('results\\'+A_name+'\\'+result,"rb")))
    
        CG_plot_all(results,pickle.load(open('results\\'+A_name+'.p',"rb")))





def CG_plot_part(results,which,title):
    fs = 18
    fst = 18
    fsticks = 13
    if which in ['TimeSetup','TimeCG','TimeOverall']:
        for result in results:
            if which == 'TimeOverall':
                plotime = result['TimeSetup']+result['TimeCG']
            else:
                plotime = result[which]

            # #optional marker for the last iteration
            # plt.scatter(plotime[-1],
            #             0,
            #             color=result['Colour'],
            #             marker='^',
            #             edgecolors='black',
            #             s=100)
            
            plt.hist(plotime,
                     color=result['Colour'],
                     alpha=0.75,
                     label='$U_{\mathrm{'+result['Prolongation'] #result['DeflationMode'] + ' '+ 
                        + '}}$, $r='+str(result['Dimension'])
                        #+ ', ω='+str(result['RelaxationFactor'])
                        + '$')
        plt.xlabel('Computation time in seconds',fontsize=fs)
        plt.ylabel('Percentage',fontsize=fs) # Number of occurrences
        plt.xticks(fontsize=fsticks)
        plt.yticks(fontsize=fsticks)
    
    elif which in ['Residual','Error']:
        maxiter = 0
        for result in results:
            if maxiter < len(result[which]):
                maxiter = len(result[which])
            plt.plot(result[which]/result[which][0],
                     color=result['Colour'],
                     label='$U_{\mathrm{'+result['Prolongation'] #result['DeflationMode'] + ' '+ 
                        + '}}$, $r='+str(result['Dimension'])
                        #+ ', ω='+str(result['RelaxationFactor'])
                        + '$')
        plt.xlim([0,maxiter*1.02])
        plt.ylim([10e-11,2])
        plt.yscale('log')
        plt.xlabel('Iteration',fontsize=fs)
        #plt.ylabel('relative '+ which)
        plt.ylabel("$||e_k||_A$"+" "+"$/$"+" "+"$||e_0||_A$" if which == "Error" else "$||r_k||$"+" "+"$/$"+" "+"$||r_0||$",fontsize=fs)
        plt.xticks(fontsize=fsticks)
        plt.yticks(fontsize=fsticks)
    n_str = str(result['MatrixShape'][0])
    plt.title(title+' with $A=A_{\mathtt{'+result['MatrixName']+"}}$, $n="+n_str+"$",fontsize=fst) #n equals \mathrm{dim}(A)





def CG_plot_final():
    fsl=16
    plt.show()
    # Create and print a list of available matrices
    matrixlist = [A_name for A_name in os.listdir('results') if A_name[-2:] !=".p"]
    for i, A_name in enumerate(matrixlist):
        print(i,A_name)
    
    # Choose number of matrices for testing
    usedmatrices = []
    morematrices = True

    print("Input instruction")
    print("Adding a matrix:     Index + ENTER")
    print("Adding all matrices: 'all' + ENTER")
    print("Abort:                'q'  + ENTER")
    print("Close selection:      'c'  + ENTER")
    while morematrices:

        choice = input()
        if   choice == 'q':
            return "aborted"
        elif choice == 'all':
            usedmatrices = matrixlist
            morematrices = False
            break
        elif choice == 'c':
            if len(usedmatrices) > 0:
                morematrices = False
            else:
                print("selection empty")
        else:    
            try:
                usedmatrices.append(matrixlist[int(choice)])
                print("Selection accepted")
            except Exception:
                print("Wrong input!")
            
    clear_output()
    
    # Create and print a list of available tests
    tests_one = os.listdir('results\\'+usedmatrices[0])       
    for i, resultname in enumerate(tests_one):
            print(i,resultname)        
    
    # Choose a number of desired tests
    testnumbers = []    
    moretests = True
    
    print("Input instruction")
    print("Adding data to plot: Index + ENTER")
    print("Abort:                'q'  + ENTER")
    print("Close selection:      'c'  + ENTER")
    
    while moretests:
        choice = input()
        if   choice == 'q':
            return "aborted"
        elif choice == 'c':
            if len(testnumbers) > 0:
                moretests = False
            else:
                print("selection empty")
        else:
            try:
                choice = int(choice)
                if choice in range(len(tests_one)):
                    testnumbers.append(choice)
                else:
                    print("Wrong Input! Index out of bounds")
            except Exception:
                print("Wrong Input!")
            
    clear_output()
    
    plots = ['TimeSetup','TimeCG','TimeOverall','Residual','Error']
    titles = ['Setup computation time',
              'CG computation time',
              'Overall computation time',
              'Comparison of residuals',
              'Comparison of errors']
    
    # Choose wanted output
    for i, part in enumerate(plots):
        print(str(i) + ': ' + part)
    choice1 = int(input('First property: '))
    choice2 = int(input('Second property: '))
    clear_output()
    
    # Plot
    NOM = len(usedmatrices) # Number Of Matrices
    fig = plt.figure(figsize=(18,6*NOM))
    fig.subplots_adjust(hspace=0.3,wspace=0.2)
    
    for index, A_name in enumerate(usedmatrices):
        tests = [os.listdir('results\\'+A_name)[i] for i in testnumbers]
        
        results = []
        for result in tests:
            results.append(pickle.load(open('results\\'+A_name+'\\'+result,"rb")))
        
        plt.subplot(NOM,2,2*index+1)
        CG_plot_part(results,plots[choice1],titles[choice1])
    
        plt.subplot(NOM,2,2*index+2)
        CG_plot_part(results,plots[choice2],titles[choice2])
        
        plt.legend(loc='upper right',fontsize=fsl)
    
    plt.savefig('Figure.svg',format='svg')
    plt.show()