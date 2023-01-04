function Tee = EETransMatrix(in1)
%EETRANSMATRIX
%    TEE = EETRANSMATRIX(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    05-Jul-2020 15:34:18

a2 = in1(:,20);
a3 = in1(:,21);
d1 = in1(:,13);
d2 = in1(:,14);
d3 = in1(:,15);
d4 = in1(:,16);
d5 = in1(:,17);
d6 = in1(:,18);
q1 = in1(:,1);
q2 = in1(:,2);
q3 = in1(:,3);
q4 = in1(:,4);
q5 = in1(:,5);
q6 = in1(:,6);
t2 = cos(q1);
t3 = cos(q2);
t4 = cos(q3);
t5 = cos(q4);
t6 = cos(q5);
t7 = cos(q6);
t8 = sin(q1);
t9 = sin(q2);
t10 = sin(q3);
t11 = sin(q4);
t12 = sin(q5);
t13 = sin(q6);
t14 = t3.*t4;
t15 = t2.*t6;
t16 = t3.*t10;
t17 = t4.*t9;
t18 = t2.*t12;
t19 = t6.*t8;
t20 = t9.*t10;
t21 = t8.*t12;
t22 = t8.*t20;
t23 = -t20;
t24 = t2.*t14;
t25 = t2.*t16;
t26 = t2.*t17;
t27 = t8.*t14;
t28 = t2.*t20;
t29 = t8.*t16;
t30 = t8.*t17;
t33 = t16+t17;
t31 = -t27;
t32 = t2.*t23;
t34 = t14+t23;
t35 = t5.*t33;
t36 = t11.*t33;
t37 = t25+t26;
t38 = t29+t30;
t39 = t5.*t34;
t40 = t11.*t34;
t41 = -t36;
t42 = t24+t32;
t43 = t22+t31;
t44 = t5.*t37;
t45 = t11.*t37;
t46 = t5.*t38;
t47 = t11.*t38;
t48 = t5.*t42;
t49 = t11.*t42;
t50 = -t45;
t51 = t5.*t43;
t52 = t11.*t43;
t54 = t35+t40;
t55 = t39+t41;
t53 = -t52;
t56 = t44+t49;
t57 = t47+t51;
t58 = t48+t50;
t62 = -t6.*(t45-t48);
t63 = -t12.*(t45-t48);
t65 = t6.*(t45-t48);
t59 = t46+t53;
t60 = t6.*t57;
t61 = t12.*t57;
t66 = t19+t63;
t68 = t21+t65;
t64 = -t60;
t67 = t18+t64;
Tee = reshape([-t13.*t56-t7.*t68,-t13.*t59+t7.*t67,t13.*(t36-t39)-t6.*t7.*t54,0.0,t7.*t56-t13.*t68,t7.*t59+t13.*t67,-t7.*(t36-t39)-t6.*t13.*t54,0.0,t66,-t15-t61,-t12.*t54,0.0,a3.*t24+a3.*t32-d2.*t8-d3.*t8-d4.*t8+d5.*t56-d6.*t66+a2.*t2.*t3,d6.*(t15+t61)-a3.*t22+a3.*t27+d2.*t2+d3.*t2+d4.*t2+d5.*t59+a2.*t3.*t8,d1-a2.*t9-a3.*t16-a3.*t17-d5.*(t36-t39)+d6.*t12.*t54,1.0],[4,4]);
