'''
@author: Julian Koslowski

A class to visoalize the results of KorrektorNet.
To plot your results in Matlab select matlab_plot.
To plot your results in Python select python_plot.
To create an excel files listing your errors select EXCEL
An example of the use of this class can be seen in the file Auswertung_KNN.py 
'''
import matplotlib.pyplot as plt 
import numpy as np
import matlab.engine

class Plot_KNN():
    def __init__(self,final_target_val_denorm_st,final_target_test_denorm_st,final_target_val_denorm_xd,final_target_test_denorm_xd,final_target_val_denorm_xm,final_target_test_denorm_xm,x_minus_val_manover_denorm,x_minus_test_manover_denorm,tensor_Messung_denorm_val,tensor_Messung_denorm_test,OEKF_val,OEKF_test,name_val,name_test):
        self.Zustand_name_list=('Winkel KingPin','Fy_R_DMS','Fy_L_DMS', 'Fz_R_DMS', 'Fz_L_DMS', 'Fy_A3_R_DMS', 'Fy_A3_L_DMS', 'Fz_A3_R_DMS', 'Fz_A3_L_DMS')
        self.Zustaende_name_Einheiten_list=('Winkel KingPin $\\theta$ in rad ','Querkraft F$_{y21R} $ in N','Querkraft F$_{y21L}$ in N', 'Aufstandskraft F$_{z21R}$  in N', 'Aufstandskraft F$_{z21L}$ in N', 'Querkraft F$_{y23R}$ in N', 'Querkraft F$_{y23L}$ in N', 'Aufstandskraft F$_{z23R}$ in N', 'Aufstandskraft F$_{z23L}$ in N')
        self.legende=('KorrektorNet X-Minus','Messung','PM')
        self.FT_val_st=final_target_val_denorm_st
        self.FT_test_st=final_target_test_denorm_st
        self.FT_val_xd=final_target_val_denorm_xd
        self.FT_test_xd=final_target_test_denorm_xd
        self.FT_val_xm=final_target_val_denorm_xm
        self.FT_test_xm=final_target_test_denorm_xm
        self.x_minus_val=x_minus_val_manover_denorm
        self.x_minus_test=x_minus_test_manover_denorm
        self.Messung_val=tensor_Messung_denorm_val
        self.Messung_test=tensor_Messung_denorm_test
        self.OEKF_val=OEKF_val
        self.OEKF_test=OEKF_test
        self.name_val=name_val
        self.name_test=name_test

    def matlab_plot(self,path_val,path_test)-> None:
        """
        Create a matlab plot
        Args:
            path_val (string): Path to save your valifation data
            path_test (string): Path to save your test data
        """
        for manover in range(len(self.FT_val_st)):
            #iteration through all manover
            FT_val_st=self.FT_val_st[manover]
            FT_test_st=self.FT_test_st[manover]
            FT_val_xd=self.FT_val_xd[manover]
            FT_test_xd=self.FT_test_xd[manover]
            FT_val_xm=self.FT_val_xm[manover]
            FT_test_xm=self.FT_test_xm[manover]
            x_minus_val=self.x_minus_val[manover]
            x_minus_test=self.x_minus_test[manover]
            Messung_val=self.Messung_val[manover]
            Messung_test=self.Messung_test[manover]
            OEKF_val=self.OEKF_val[manover]
            OEKF_test=self.OEKF_test[manover]
            name_val=self.name_val[manover]
            name_test=self.name_test[manover]
            for Zustaende in range(FT_val_st.shape[1]):
                #iteration through all states validation
                eng=matlab.engine.start_matlab()
                x_input_val=np.array([FT_val_st[:,Zustaende],FT_val_xd[:,Zustaende],FT_val_xm[:,Zustaende],Messung_val[:,Zustaende].detach().numpy()])
                x_input_test=np.array([FT_test_st[:,Zustaende],FT_test_xd[:,Zustaende],FT_test_xm[:,Zustaende],Messung_test[:,Zustaende].detach().numpy()])
                x_val=matlab.double(x_input_val.tolist())
                x_test=matlab.double(x_input_test.tolist())
                y_val=matlab.double(np.arange(0,FT_val_st.shape[0]/100,1/100))
                y_test=matlab.double(np.arange(0,FT_test_st.shape[0]/100,1/100))
                _=eng.plot_template(x_val,y_val,'Zeit $t$ in s',self.Zustaende_name_Einheiten_list[Zustaende],('KorrektorNet Standalone','KorrektorNet xDach','KorrektorNet xMinus','Messung'),matlab.double(1),rf'{path_val}/val_{name_val}_{ self.Zustand_name_list[Zustaende]}_KNN')
                _=eng.plot_template(x_test,y_test,'Zeit $t$ in s',self.Zustaende_name_Einheiten_list[Zustaende],('KorrektorNet Standalone','KorrektorNet xDach','KorrektorNet xMinus','Messung'),matlab.double(1),rf'{path_test}/test_{name_test}_{ self.Zustand_name_list[Zustaende]}_KNN')
            for Zustaende in range(FT_val_xd.shape[1]):
                #iteration through all test validation
                eng=matlab.engine.start_matlab()
                x_input_val=np.array([FT_val_xd[:,Zustaende],OEKF_val[Zustaende,:],x_minus_val[:,Zustaende+4],Messung_val[:,Zustaende].detach().numpy()])
                x_input_test=np.array([FT_test_xd[:,Zustaende],OEKF_test[Zustaende,:],x_minus_test[:,Zustaende+4],Messung_test[:,Zustaende].detach().numpy()])
                x_val=matlab.double(x_input_val.tolist())
                x_test=matlab.double(x_input_test.tolist())
                y_val=matlab.double(np.arange(0,FT_val_st.shape[0]/100,1/100))
                y_test=matlab.double(np.arange(0,FT_test_st.shape[0]/100,1/100))
                _=eng.plot_template(x_val,y_val,'Zeit $t$ in s',self.Zustaende_name_Einheiten_list[Zustaende],('KorrektorNet xDach','EKF','PM','Messung'),matlab.double(1),rf'{path_val}/val_{name_val}_{self.Zustand_name_list[Zustaende]}_PM')
                _=eng.plot_template(x_test,y_test,'Zeit $t$ in s',self.Zustaende_name_Einheiten_list[Zustaende],('KorrektorNet xDach','EKF','PM','Messung'),matlab.double(1),rf'{path_test}/test_{name_test}_{ self.Zustand_name_list[Zustaende]}_PM')
    def python_plot(self,path_val,path_test):
        """
        Create a plot
        Args:
            path_val (string): Path to save your valifation data
            path_test (string): Path to save your test data
        """
        for manover in range(len(self.FT_val_st)):
            #iteration through all manover
            FT_val=self.FT_val[manover]
            FT_test=self.FT_test[manover]
            x_minus_val=self.x_minus_val[manover]
            x_minus_test=self.x_minus_test[manover]
            Messung_val=self.Messung_val[manover]
            Messung_test=self.Messung_test[manover]
            OEKF=self.OEKF[manover]
            name_val=self.name_val[manover]
            name_test=self.name_test[manover]
            for Zustaende in range(FT_val.shape[1]):
                #iteration through all states validation
                plt.figure(figsize=(19.20,9.83))
                plt.title(f'{name_val}_val_{self.Zustand_name_list[Zustaende]}')
                plt.plot(FT_val[:,Zustaende],label='X_korr')
                plt.plot(Messung_val[:,Zustaende].detach().numpy(),label='Messung')
                plt.plot(x_minus_val[:,Zustaende+4],label='PM')
                plt.legend()
                plt.savefig(f'{path_val}_val_{name_val}.png',dpi=600)
                plt.savefig(f'{path_val}_val_{name_val}.pdf',dpi=600)
                plt.close()
                plt.figure(figsize=(19.20,9.83))
                plt.title(f'{name_test}_test_{self.Zustand_name_list[Zustaende]}')
                plt.plot(FT_test[:,Zustaende],label='X_korr')
                plt.plot(Messung_test[:,Zustaende].detach().numpy(),label='Messung')
                plt.plot(x_minus_test[:,Zustaende+4],label='PM')
                plt.legend()
                plt.savefig(f'{path_test}_test_{name_test}.png',dpi=600)
                plt.savefig(f'{path_test}_test_{name_test}.pdf',dpi=600)
                plt.close()
    def Excel(self,path_val,path_test):
        """
        Create an excel-file
        Args:
            path_val (_type_): Path to save your test data
            path_test (_type_): Path to save your test data
        """
        for manover in range(len(self.FT_val_st)):
            #iteration through all states validation
            FT_val_st=self.FT_val_st[manover]
            FT_test_st=self.FT_test_st[manover]
            FT_val_xd=self.FT_val_xd[manover]
            FT_test_xd=self.FT_test_xd[manover]
            FT_val_xm=self.FT_val_xm[manover]
            FT_test_xm=self.FT_test_xm[manover]
            x_minus_val=self.x_minus_val[manover]
            x_minus_test=self.x_minus_test[manover]
            Messung_val=self.Messung_val[manover]
            Messung_test=self.Messung_test[manover]
            OEKF_val=self.OEKF_val[manover]
            OEKF_test=self.OEKF_test[manover]
            name_val=self.name_val[manover]
            name_test=self.name_test[manover]
            for Zustaende in range(FT_val_st.shape[1]):
                #iteration through all test validation
                EXC_val=excel(path_val)
                EXC_test=excel(path_test)
                #compute the MSE
                error_Zustand_val_st=mean_squared_error(FT_val_st[:,Zustaende],Messung_val[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_test_st=mean_squared_error(FT_test_st[:,Zustaende],Messung_test[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_val_xd=mean_squared_error(FT_val_xd[:,Zustaende],Messung_val[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_test_xd=mean_squared_error(FT_test_xd[:,Zustaende],Messung_test[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_val_xm=mean_squared_error(FT_val_xm[:,Zustaende],Messung_val[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_test_xm=mean_squared_error(FT_test_xm[:,Zustaende],Messung_test[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_val_pm=mean_squared_error(x_minus_val[:,Zustaende+4],Messung_val[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_test_pm=mean_squared_error(x_minus_test[:,Zustaende+4],Messung_test[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_val_ekf=mean_squared_error(OEKF_val[Zustaende,:],Messung_val[:,Zustaende].detach().numpy(),squared=False)
                error_Zustand_test_ekf=mean_squared_error(OEKF_test[Zustaende,:],Messung_test[:,Zustaende].detach().numpy(),squared=False)
                #safe into an excel-file
                EXC_val.write(error_Zustand_val_st,Fahrt=name_val,Zustand=Zustaende,Verfahren='Standalone')
                EXC_test.write(error_Zustand_test_st,Fahrt=name_test,Zustand=Zustaende,Verfahren='Standalone')
                EXC_val.write(error_Zustand_val_xd,Fahrt=name_val,Zustand=Zustaende,Verfahren='x-Dach')
                EXC_test.write(error_Zustand_test_xd,Fahrt=name_test,Zustand=Zustaende,Verfahren='x-Dach')
                EXC_val.write(error_Zustand_val_xm,Fahrt=name_val,Zustand=Zustaende,Verfahren='x-Minus')
                EXC_test.write(error_Zustand_test_xm,Fahrt=name_test,Zustand=Zustaende,Verfahren='x-Minus')
                EXC_val.write(error_Zustand_val_pm,Fahrt=name_val,Zustand=Zustaende,Verfahren='PML')
                EXC_test.write(error_Zustand_test_pm,Fahrt=name_test,Zustand=Zustaende,Verfahren='PML')
                EXC_val.write(error_Zustand_val_ekf,Fahrt=name_val,Zustand=Zustaende,Verfahren='EKF')
                EXC_test.write(error_Zustand_test_ekf,Fahrt=name_test,Zustand=Zustaende,Verfahren='EKF')


