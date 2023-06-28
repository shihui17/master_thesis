q_integrate = dlmread('C:\Codes\master_thesis\discrete_int_q.txt');
time_integrate = dlmread('C:\Codes\master_thesis\t_int_trace.txt');
data = struct('x', time_integrate, 'y', q_integrate);
save('data.mat', 'data')