force_prof = dlmread('C:\Matlab_workspace\Matlab Plot Vorlage\matlab_plot\momentum_opt\mean_force.txt');
iter = linspace(0, 21, 22);
force_prof_max = struct('x', iter, 'y', force_prof);
save('C:\Matlab_workspace\Matlab Plot Vorlage\matlab_plot\force_prof_max', 'force_prof_max')