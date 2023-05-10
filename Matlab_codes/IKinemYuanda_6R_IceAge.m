function [ q ] = IKinemYuanda_6R_IceAge( robot, XEE,  config, opt )


%IKinemYuanda_6R is identical to IKinemUR5




%% Inputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% config = [1 1 1 1 -1 1];
% config = [1 1 1];

% Option vector for all 6 joints
% % config = [1 -1 -1 0 1 0];  % config 1
% % config = [-1 1 -1 0 1 0];  % config 2
% % config = [1 -1 1 0 1 0];   % config 3
% % config = [-1 1 -1 0 1 0];  % config 4
% % config = [1 -1 -1 0 -1 0]; % config 5
% % config = [-1 1 -1 0 -1 0]; % config 6
% % config = [1 -1 1 0 -1 0];  % config 7
% % config = [-1 1 -1 0 -1 0]; % config 8

% Real (possible) option vector

% config = [1 1 1];

% config = [1 1 1];  % config 1
% config = [1 1 -1];  % config 2
% config = [1 -1 1];   % config 3
% config = [1 -1 -1];  % config 4
% config = [-1 1 1]; % config 5
% config = [-1 1 -1]; % config 6
% config = [-1 -1 1];  % config 7
% config = [-1 -1 -1]; % config 8

switch opt.InputType
        
    case 'RPYangles'
        x=XEE(1);
        y=XEE(2);
        z=XEE(3);
        alpha=XEE(4);
        beta=XEE(5);
        gamma=XEE(6);
        
        [ REE ] = RPY( alpha, beta, gamma);

        T_0_6 = [REE [x; y; z]; 0 0 0 1];
        
    case 'HomTransMatrix'
        T_0_6 = XEE;
        
        x = T_0_6(1,4);
        y = T_0_6(2,4);
        z = T_0_6(3,4);
        REE = T_0_6(1:3,1:3);
        
end


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a1 = robot.Links(1).a;
a2 = robot.Links(2).a;
a3 = robot.Links(3).a;

d1 = robot.Links(1).d;
d2 = robot.Links(2).d;
d3 = robot.Links(3).d;
d4 = robot.Links(4).d;
d5 = robot.Links(5).d;
d6 = robot.Links(6).d;

% A1 = robot.Links(1).A(q1);
% A2 = robot.Links(2).A(q2);
% A3 = robot.Links(3).A(q3);
% A4 = robot.Links(4).A(q4);
% A5 = robot.Links(5).A(q5);
% A6 = robot.Links(6).A(q6);


%% q1 calculation

P_0_6 = T_0_6(1:3,4);
ez_06 = T_0_6(1:3,3);
% P_0_5 = P_0_6 - d6*ez_06;
P_0_5 = P_0_6 - abs(d6)*ez_06;

if (P_0_5(1)^2 + P_0_5(2)^2) < (d2+d3+d4)^2
    display('Point outside of the workspace. q1 is not possible! ');
    q = [];
    return
end
    
q1 = atan2((d2+d3+d4), -config(1)*sqrt(P_0_5(1)^2 + P_0_5(2)^2 - (d2+d3+d4)^2)) - atan2(P_0_5(2),-P_0_5(1));


%% q5, q6 calculation

A1 = robot.Links(1).A(q1);
T_1_6 = inv(A1) * T_0_6;

% q5 = config(5)*acos(T_1_6(3,3));

%-------------------------------------%

% K_1 = T_1_6(3,2) + T_1_6(3,3);
% K_a = sin(q6);
% K_b = 1;
% 
% q5 = atan2(K_1, config(5)*sqrt( K_a^2 + K_b^2 - K_1^2 )) - atan2(K_b, K_a);

% K_a = T_1_6(3,3);
% K_b = config(5)*sqrt(1-T_1_6(3,3)^2);
% 
% q5 = atan2(K_b, K_a);
%-------------------------------------%


P_1_6 = T_1_6(1:3,4);

if ((P_1_6(3)-d2-d3-d4)/(d6)) > 1  % 0.999999?
    display('Point outside of the workspace. q5 is not possible! ');
    q = [];
    return
end

q5 = config(3)*acos( (P_1_6(3)-d2-d3-d4)/(d6) );
% q5 = config(5)*acos( (P_1_6(3)-d2-d3-d4)/(d6) );

%------------------%
% If q5==0 then this is a singularity!!!!!

if abs(q5)<1E-3
    q6 = 0;
    warning('q5 = 0. Singualr pose!!!!')
else
    q6 = atan2( config(3)*T_1_6(3,2), config(3)*T_1_6(3,1));
%     q6 = atan2( config(3)*T_1_6(3,2), -config(3)*T_1_6(3,1));
end

% q6 = atan2( config(3)*T_1_6(3,2), -config(3)*T_1_6(3,1));
% q6 = atan2( config(3)*T_1_6(3,2), -config(3)*T_1_6(3,1));
%-------------------------------------%
% q6 = atan2( config(5)*T_1_6(3,2), -config(5)*T_1_6(3,1));


%% q3 calculation

A5 = robot.Links(5).A(q5);
A6 = robot.Links(6).A(q6);
T_1_4 = inv(A1) * T_0_6 * inv(A5*A6);
P_1_4 = T_1_4(1:3,4);

K_temp_1 = (( P_1_4(1)^2 + P_1_4(2)^2 - a2^2 - a3^2 )/(2 * a2 * a3));

if K_temp_1 > (1+1E-5)  % 0.999999?
    display('Point outside of the workspace. q3 is not possible! ');
    q = [];
    return
end

if K_temp_1 > 0.999  % 0.999999?
    K_temp_1 = 1;
end


q3 = config(2)*acos(K_temp_1);


% if (( P_1_4(1)^2 + P_1_4(2)^2 - a2^2 - a3^2 )/(2 * a2 * a3)) > (1+1E-6)  % 0.999999?
%     display('Point outside of the workspace. q3 is not possible! ');
%     q = [];
%     return
% end
% 
% if (( P_1_4(1)^2 + P_1_4(2)^2 - a2^2 - a3^2 )/(2 * a2 * a3)) > (1+1E-6)  % 0.999999?
%     display('Point outside of the workspace. q3 is not possible! ');
%     q = [];
%     return
% end
% 
% 
% q3 = config(2)*acos(( P_1_4(1)^2 + P_1_4(2)^2 - a2^2 - a3^2 )/(2 * a2 * a3));

% % % q3 = config(3)*acos(( P_1_4(1)^2 + P_1_4(2)^2 - a2^2 - a3^2 )/(2 * a2 * a3));


%% q2 calculation

K_1 = P_1_4(1) + P_1_4(2);
K_a = -a3*sin(q3) + a3*cos(q3) + a2;
K_b = a3*sin(q3) + a3*cos(q3) + a2;

factor = [];
% if ((abs(P_1_4(1)) - abs(P_1_4(2))) > 0 )
%     factor = -1;
% end

if (P_1_4(1) >= P_1_4(2))
    factor = 1;
else
    factor = -1;
end


% factor = -1;
% factor
q2 = atan2(K_1, factor*sqrt( K_a^2 + K_b^2 - K_1^2 )) - atan2(K_b, K_a);
% q2 = atan2(K_1, -config(1)*sqrt( K_a^2 + K_b^2 - K_1^2 )) - atan2(K_b, K_a);
% % % % % q2 = atan2(K_1, config(2)*sqrt( K_a^2 + K_b^2 - K_1^2 )) - atan2(K_b, K_a);

% q2 = asin(K_1 / sqrt( K_a^2 + K_b^2 )) - atan2(K_b, K_a);

%-------------------------------------------------------------%
% disp('P_1_4(1) = ')
% P_1_4(1)
% disp('P_1_4(2) = ')
% P_1_4(2)
% disp('K_1 = ')
% K_1
% disp('K_1^2 = ')
% K_1^2
% disp('K_a^2 + K_b^2 = ')
% K_a^2 + K_b^2
% disp('sqrt( K_a^2 + K_b^2 - K_1^2 )= ')
% sqrt( K_a^2 + K_b^2 - K_1^2 )
% disp('K_b = ')
% K_b
% disp('K_a = ')
% K_a
% % ang1 = atan2(K_1, -config(1)*sqrt( K_a^2 + K_b^2 - K_1^2 ));
% ang1 = atan2(K_1, factor*sqrt( K_a^2 + K_b^2 - K_1^2 ));
% ang2 = - atan2(K_b, K_a);
% disp('ang1 = ')
% ang1*180/pi
% disp('ang2 = ')
% ang2*180/pi
% disp('************************')
% disp('P_1_4(1) = ')
% P_1_4(1)
% disp('P_1_4(2) = ')
% P_1_4(2)
% disp('x-y =')
% abs(P_1_4(1)) - abs(P_1_4(2))
%-------------------------------------------------------------%


% disp('K_1 = ')
% K_1
% disp('sqrt( K_a^2 + K_b^2 ) = ')
% sqrt( K_a^2 + K_b^2 )
% disp('K_b = ')
% K_b
% disp('K_a = ')
% K_a
% ang1 = asin(K_1 / sqrt( K_a^2 + K_b^2 ));
% ang2 = - atan2(K_b, K_a);
% disp('ang1 = ')
% ang1*180/pi
% disp('ang2 = ')
% ang2*180/pi


%% q4 calculation

A2 = robot.Links(2).A(q2);
A3 = robot.Links(3).A(q3);  
T_3_4 = inv(A1*A2*A3) * T_0_6 * inv(A5*A6);

q4 = atan2(T_3_4(2,1), T_3_4(1,1));


%%

q = [q1 q2 q3 q4 q5 q6]';




end


function [ R ] = RPY( alpha, beta, gamma )

% Option f?r deg und rad hinzuf?gen
% strcmp(opt, 'rad')

%     alpha = alpha*pi/180;
%     beta = beta*pi/180;
%     gamma = gamma*pi/180;


% 
% R = [ cos(gamma)*cos(beta)  cos(gamma)*sin(beta)*sin(alpha)-sin()

Rx = rotx(alpha);
Ry = roty(beta);
Rz = rotz(gamma);
R = Rz*Ry*Rx;

end