'''import random as rd
for k in range (20):
    mydict = {}

    for i in range(8):
        thislist = []
        numeros = rd.sample(range(1, 6), 5)
        for j in range(5):
            tup = ("machine_{m}".format(m = numeros[j]), 1)
            thislist.append(tup)
        mydict["job_{num}".format(num = i + 1)] = thislist
    print ("Exp_{nu}= ".format(nu = k + 1) + str(mydict))'''

Exp_1= {'job_1': [('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1)], 'job_2': [('machine_2', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1), ('machine_5', 1)], 'job_3': [('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1), ('machine_1', 1)], 'job_4': [('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1)], 'job_5': [('machine_2', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1)], 'job_6': [('machine_2', 1), ('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_5', 1)], 'job_7': [('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_1', 1), ('machine_5', 1)], 'job_8': [('machine_5', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1), ('machine_2', 1)]}
Exp_2= {'job_1': [('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1), ('machine_5', 1)], 'job_2': [('machine_5', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1)], 'job_3': [('machine_3', 1), ('machine_4', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1)], 'job_4': [('machine_5', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1)], 'job_5': [('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)], 'job_6': [('machine_1', 1), ('machine_5', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1)], 'job_7': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1)], 'job_8': [('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_2', 1), ('machine_3', 1)]}
Exp_3= {'job_1': [('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_4', 1)], 'job_2': [('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1)], 'job_3': [('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_1', 1)], 'job_4': [('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1)], 'job_5': [('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1), ('machine_3', 1)], 'job_6': [('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_4', 1)], 'job_7': [('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1)], 'job_8': [('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1)]}
Exp_4= {'job_1': [('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1)], 'job_2': [('machine_5', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1)], 'job_3': [('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1)], 'job_4': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1)], 'job_5': [('machine_5', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_2', 1)], 'job_6': [('machine_1', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1)], 'job_7': [('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1)], 'job_8': [('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)]}
Exp_5= {'job_1': [('machine_4', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1)], 'job_2': [('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_1', 1), ('machine_5', 1)], 'job_3': [('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_4', 1)], 'job_4': [('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1)], 'job_5': [('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1)], 'job_6': [('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_3', 1)], 'job_7': [('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1)], 'job_8': [('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1)]}
Exp_6= {'job_1': [('machine_2', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_5', 1)], 'job_2': [('machine_4', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1)], 'job_3': [('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1)], 'job_4': [('machine_4', 1), ('machine_1', 1), ('machine_2', 1), ('machine_5', 1), ('machine_3', 1)], 'job_5': [('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1), ('machine_5', 1)], 'job_6': [('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_4', 1)], 'job_7': [('machine_1', 1), ('machine_5', 1), ('machine_2', 1), ('machine_3', 1), ('machine_4', 1)], 'job_8': [('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)]}
Exp_7= {'job_1': [('machine_4', 1), ('machine_2', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1)], 'job_2': [('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1)], 'job_3': [('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1)], 'job_4': [('machine_5', 1), ('machine_2', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1)], 'job_5': [('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_1', 1), ('machine_5', 1)], 'job_6': [('machine_5', 1), ('machine_2', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1)], 'job_7': [('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1)], 'job_8': [('machine_2', 1), ('machine_5', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1)]}
Exp_8= {'job_1': [('machine_2', 1), ('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1)], 'job_2': [('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1)], 'job_3': [('machine_4', 1), ('machine_1', 1), ('machine_2', 1), ('machine_5', 1), ('machine_3', 1)], 'job_4': [('machine_4', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1)], 'job_5': [('machine_1', 1), ('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1)], 'job_6': [('machine_1', 1), ('machine_2', 1), ('machine_5', 1), ('machine_4', 1), ('machine_3', 1)], 'job_7': [('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1)], 'job_8': [('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1)]}
Exp_9= {'job_1': [('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_4', 1)], 'job_2': [('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1)], 'job_3': [('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1)], 'job_4': [('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1)], 'job_5': [('machine_3', 1), ('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1)], 'job_6': [('machine_3', 1), ('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_2', 1)], 'job_7': [('machine_3', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1), ('machine_4', 1)], 'job_8': [('machine_5', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1), ('machine_2', 1)]}
Exp_10= {'job_1': [('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1)], 'job_2': [('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)], 'job_3': [('machine_3', 1), ('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_2', 1)], 'job_4': [('machine_2', 1), ('machine_4', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1)], 'job_5': [('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1), ('machine_4', 1)], 'job_6': [('machine_5', 1), ('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_3', 1)], 'job_7': [('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1), ('machine_1', 1)], 'job_8': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1)]}
Exp_11= {'job_1': [('machine_4', 1), ('machine_3', 1), ('machine_1', 1), ('machine_2', 1), ('machine_5', 1)], 'job_2': [('machine_3', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1)], 'job_3': [('machine_2', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_5', 1)], 'job_4': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1)], 'job_5': [('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_4', 1)], 'job_6': [('machine_5', 1), ('machine_2', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1)], 'job_7': [('machine_1', 1), ('machine_5', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1)], 'job_8': [('machine_3', 1), ('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_1', 1)]}
Exp_12= {'job_1': [('machine_2', 1), ('machine_5', 1), ('machine_4', 1), ('machine_1', 1), ('machine_3', 1)], 'job_2': [('machine_2', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1)], 'job_3': [('machine_4', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1), ('machine_3', 1)], 'job_4': [('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1), ('machine_4', 1)], 'job_5': [('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1), ('machine_3', 1)], 'job_6': [('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1)], 'job_7': [('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1)], 'job_8': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1)]}
Exp_13= {'job_1': [('machine_3', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_4', 1)], 'job_2': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1)], 'job_3': [('machine_3', 1), ('machine_2', 1), ('machine_5', 1), ('machine_4', 1), ('machine_1', 1)], 'job_4': [('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1)], 'job_5': [('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_4', 1)], 'job_6': [('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1)], 'job_7': [('machine_1', 1), ('machine_2', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1)], 'job_8': [('machine_2', 1), ('machine_5', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1)]}
Exp_14= {'job_1': [('machine_5', 1), ('machine_2', 1), ('machine_3', 1), ('machine_1', 1), ('machine_4', 1)], 'job_2': [('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)], 'job_3': [('machine_5', 1), ('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_3', 1)], 'job_4': [('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)], 'job_5': [('machine_1', 1), ('machine_5', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1)], 'job_6': [('machine_4', 1), ('machine_3', 1), ('machine_1', 1), ('machine_5', 1), ('machine_2', 1)], 'job_7': [('machine_3', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1), ('machine_1', 1)], 'job_8': [('machine_4', 1), ('machine_2', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1)]}
Exp_15= {'job_1': [('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1)], 'job_2': [('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1)], 'job_3': [('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1)], 'job_4': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1)], 'job_5': [('machine_3', 1), ('machine_1', 1), ('machine_2', 1), ('machine_5', 1), ('machine_4', 1)], 'job_6': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1)], 'job_7': [('machine_4', 1), ('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1)], 'job_8': [('machine_2', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1)]}
Exp_16= {'job_1': [('machine_1', 1), ('machine_2', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)], 'job_2': [('machine_1', 1), ('machine_5', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1)], 'job_3': [('machine_3', 1), ('machine_4', 1), ('machine_2', 1), ('machine_5', 1), ('machine_1', 1)], 'job_4': [('machine_3', 1), ('machine_2', 1), ('machine_5', 1), ('machine_1', 1), ('machine_4', 1)], 'job_5': [('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1)], 'job_6': [('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_1', 1)], 'job_7': [('machine_2', 1), ('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)], 'job_8': [('machine_2', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1)]}
Exp_17= {'job_1': [('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)], 'job_2': [('machine_2', 1), ('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1)], 'job_3': [('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1), ('machine_4', 1)], 'job_4': [('machine_5', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_2', 1)], 'job_5': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1)], 'job_6': [('machine_2', 1), ('machine_5', 1), ('machine_4', 1), ('machine_1', 1), ('machine_3', 1)], 'job_7': [('machine_2', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1), ('machine_1', 1)], 'job_8': [('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1)]}
Exp_18= {'job_1': [('machine_3', 1), ('machine_4', 1), ('machine_2', 1), ('machine_1', 1), ('machine_5', 1)], 'job_2': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1)], 'job_3': [('machine_5', 1), ('machine_4', 1), ('machine_2', 1), ('machine_3', 1), ('machine_1', 1)], 'job_4': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1)], 'job_5': [('machine_4', 1), ('machine_1', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)], 'job_6': [('machine_1', 1), ('machine_3', 1), ('machine_5', 1), ('machine_4', 1), ('machine_2', 1)], 'job_7': [('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_5', 1)], 'job_8': [('machine_4', 1), ('machine_5', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1)]}
Exp_19= {'job_1': [('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1), ('machine_3', 1)], 'job_2': [('machine_2', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1), ('machine_1', 1)], 'job_3': [('machine_2', 1), ('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_5', 1)], 'job_4': [('machine_3', 1), ('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1)], 'job_5': [('machine_2', 1), ('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_1', 1)], 'job_6': [('machine_1', 1), ('machine_3', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1)], 'job_7': [('machine_2', 1), ('machine_1', 1), ('machine_4', 1), ('machine_3', 1), ('machine_5', 1)], 'job_8': [('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1), ('machine_2', 1)]}
Exp_20= {'job_1': [('machine_1', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_3', 1)], 'job_2': [('machine_1', 1), ('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_3', 1)], 'job_3': [('machine_5', 1), ('machine_4', 1), ('machine_3', 1), ('machine_2', 1), ('machine_1', 1)], 'job_4': [('machine_2', 1), ('machine_1', 1), ('machine_5', 1), ('machine_3', 1), ('machine_4', 1)], 'job_5': [('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_2', 1), ('machine_3', 1)], 'job_6': [('machine_3', 1), ('machine_1', 1), ('machine_4', 1), ('machine_5', 1), ('machine_2', 1)], 'job_7': [('machine_2', 1), ('machine_4', 1), ('machine_5', 1), ('machine_1', 1), ('machine_3', 1)], 'job_8': [('machine_3', 1), ('machine_5', 1), ('machine_2', 1), ('machine_4', 1), ('machine_1', 1)]}

Exp_List = [Exp_1, Exp_2, Exp_3, Exp_4, Exp_5, Exp_6, Exp_7, Exp_8, Exp_9, Exp_10,
            Exp_11, Exp_12, Exp_13, Exp_14, Exp_15, Exp_16, Exp_17, Exp_18, Exp_19, Exp_20]