function JTool = calcJacobianTCP(in1)
%JACOBIANMATRIXTOOL
%    JTOOL = JACOBIANMATRIXTOOL(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    05-Jul-2020 15:34:14

T_n_TCP_1_4 = in1(:,40);
T_n_TCP_2_4 = in1(:,41);
T_n_TCP_3_4 = in1(:,42);
a2 = in1(:,20);
a3 = in1(:,21);
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
t14 = q2+q3;
t19 = -q1;
t20 = -q5;
t21 = -q6;
t15 = a2.*t9;
t16 = cos(t14);
t17 = q4+t14;
t18 = sin(t14);
t27 = -t8;
t22 = cos(t17);
t23 = q1+t17;
t24 = q5+t17;
t25 = q6+t17;
t26 = sin(t17);
t28 = a3.*t16;
t29 = a3.*t18;
t40 = t17+t19;
t41 = t17+t20;
t42 = t17+t21;
t30 = cos(t24);
t31 = cos(t25);
t32 = q6+t24;
t33 = sin(t24);
t34 = sin(t25);
t36 = d5.*t22;
t37 = d5.*t26;
t39 = -t28;
t44 = cos(t41);
t45 = cos(t42);
t47 = t21+t24;
t48 = t20+t25;
t49 = sin(t41);
t50 = sin(t42);
t68 = t21+t41;
t35 = sin(t32);
t38 = cos(t32);
t43 = -t36;
t46 = -t37;
t51 = cos(t47);
t52 = cos(t48);
t53 = sin(t47);
t54 = sin(t48);
t55 = (T_n_TCP_1_4.*t31)./2.0;
t56 = (T_n_TCP_2_4.*t31)./2.0;
t57 = (T_n_TCP_3_4.*t30)./2.0;
t58 = (d6.*t30)./2.0;
t59 = (T_n_TCP_1_4.*t34)./2.0;
t60 = (T_n_TCP_2_4.*t34)./2.0;
t61 = (T_n_TCP_3_4.*t33)./2.0;
t62 = (d6.*t33)./2.0;
t75 = cos(t68);
t77 = sin(t68);
t78 = (T_n_TCP_1_4.*t45)./2.0;
t79 = (T_n_TCP_2_4.*t45)./2.0;
t80 = (T_n_TCP_3_4.*t44)./2.0;
t81 = (d6.*t44)./2.0;
t82 = (T_n_TCP_1_4.*t50)./2.0;
t83 = (T_n_TCP_2_4.*t50)./2.0;
t84 = (T_n_TCP_3_4.*t49)./2.0;
t85 = (d6.*t49)./2.0;
t63 = -t55;
t64 = -t56;
t65 = -t57;
t66 = -t60;
t67 = -t61;
t69 = (T_n_TCP_1_4.*t38)./4.0;
t70 = (T_n_TCP_2_4.*t38)./4.0;
t71 = (T_n_TCP_1_4.*t35)./4.0;
t72 = (T_n_TCP_2_4.*t35)./4.0;
t86 = (T_n_TCP_1_4.*t51)./4.0;
t87 = (T_n_TCP_1_4.*t52)./4.0;
t88 = (T_n_TCP_2_4.*t51)./4.0;
t89 = (T_n_TCP_2_4.*t52)./4.0;
t90 = (T_n_TCP_1_4.*t53)./4.0;
t91 = (T_n_TCP_1_4.*t54)./4.0;
t92 = (T_n_TCP_2_4.*t53)./4.0;
t93 = (T_n_TCP_2_4.*t54)./4.0;
t94 = -t79;
t95 = -t81;
t96 = -t82;
t97 = -t83;
t98 = -t85;
t103 = (T_n_TCP_1_4.*t75)./4.0;
t104 = (T_n_TCP_2_4.*t75)./4.0;
t105 = (T_n_TCP_1_4.*t77)./4.0;
t106 = (T_n_TCP_2_4.*t77)./4.0;
t73 = -t69;
t74 = -t70;
t76 = -t72;
t99 = -t86;
t100 = -t87;
t101 = -t89;
t102 = -t93;
t107 = -t103;
t108 = t43+t58+t59+t64+t65+t71+t74+t80+t88+t90+t91+t94+t95+t96+t101+t104+t105;
t109 = t29+t108;
t110 = t15+t109;
JTool = reshape([-d2.*t2-d3.*t2-d4.*t2+T_n_TCP_3_4.*t2.*t6+a2.*t3.*t27-d6.*t2.*t6-T_n_TCP_1_4.*t2.*t7.*t12-T_n_TCP_2_4.*t2.*t12.*t13+a3.*t8.*t9.*t10+a3.*t3.*t4.*t27+d5.*t8.*t9.*t10.*t11+d5.*t3.*t4.*t11.*t27+d5.*t3.*t5.*t10.*t27+d5.*t4.*t5.*t9.*t27+d6.*t3.*t4.*t5.*t8.*t12+d6.*t3.*t10.*t11.*t12.*t27+d6.*t4.*t9.*t11.*t12.*t27+d6.*t5.*t9.*t10.*t12.*t27+T_n_TCP_1_4.*t3.*t4.*t8.*t11.*t13+T_n_TCP_1_4.*t3.*t5.*t8.*t10.*t13+T_n_TCP_1_4.*t4.*t5.*t8.*t9.*t13+T_n_TCP_2_4.*t7.*t8.*t9.*t10.*t11+T_n_TCP_2_4.*t3.*t4.*t7.*t11.*t27+T_n_TCP_2_4.*t3.*t5.*t7.*t10.*t27+T_n_TCP_2_4.*t4.*t5.*t7.*t9.*t27+T_n_TCP_3_4.*t3.*t8.*t10.*t11.*t12+T_n_TCP_3_4.*t4.*t8.*t9.*t11.*t12+T_n_TCP_3_4.*t5.*t8.*t9.*t10.*t12+T_n_TCP_1_4.*t9.*t10.*t11.*t13.*t27+T_n_TCP_3_4.*t3.*t4.*t5.*t12.*t27+T_n_TCP_1_4.*t3.*t6.*t7.*t8.*t10.*t11+T_n_TCP_1_4.*t4.*t6.*t7.*t8.*t9.*t11+T_n_TCP_1_4.*t5.*t6.*t7.*t8.*t9.*t10+T_n_TCP_1_4.*t3.*t4.*t5.*t6.*t7.*t27+T_n_TCP_2_4.*t3.*t6.*t8.*t10.*t11.*t13+T_n_TCP_2_4.*t4.*t6.*t8.*t9.*t11.*t13+T_n_TCP_2_4.*t5.*t6.*t8.*t9.*t10.*t13+T_n_TCP_2_4.*t3.*t4.*t5.*t6.*t13.*t27,d2.*t27+d3.*t27+d4.*t27+T_n_TCP_3_4.*t6.*t8+a2.*t2.*t3+d6.*t6.*t27+T_n_TCP_1_4.*t7.*t12.*t27+T_n_TCP_2_4.*t12.*t13.*t27+a3.*t2.*t3.*t4-a3.*t2.*t9.*t10+d5.*t2.*t3.*t4.*t11+d5.*t2.*t3.*t5.*t10+d5.*t2.*t4.*t5.*t9-d5.*t2.*t9.*t10.*t11-d6.*t2.*t3.*t4.*t5.*t12+d6.*t2.*t3.*t10.*t11.*t12+d6.*t2.*t4.*t9.*t11.*t12+d6.*t2.*t5.*t9.*t10.*t12-T_n_TCP_1_4.*t2.*t3.*t4.*t11.*t13-T_n_TCP_1_4.*t2.*t3.*t5.*t10.*t13-T_n_TCP_1_4.*t2.*t4.*t5.*t9.*t13+T_n_TCP_2_4.*t2.*t3.*t4.*t7.*t11+T_n_TCP_2_4.*t2.*t3.*t5.*t7.*t10+T_n_TCP_2_4.*t2.*t4.*t5.*t7.*t9+T_n_TCP_1_4.*t2.*t9.*t10.*t11.*t13+T_n_TCP_3_4.*t2.*t3.*t4.*t5.*t12-T_n_TCP_2_4.*t2.*t7.*t9.*t10.*t11-T_n_TCP_3_4.*t2.*t3.*t10.*t11.*t12-T_n_TCP_3_4.*t2.*t4.*t9.*t11.*t12-T_n_TCP_3_4.*t2.*t5.*t9.*t10.*t12+T_n_TCP_1_4.*t2.*t3.*t4.*t5.*t6.*t7-T_n_TCP_1_4.*t2.*t3.*t6.*t7.*t10.*t11-T_n_TCP_1_4.*t2.*t4.*t6.*t7.*t9.*t11-T_n_TCP_1_4.*t2.*t5.*t6.*t7.*t9.*t10+T_n_TCP_2_4.*t2.*t3.*t4.*t5.*t6.*t13-T_n_TCP_2_4.*t2.*t3.*t6.*t10.*t11.*t13-T_n_TCP_2_4.*t2.*t4.*t6.*t9.*t11.*t13-T_n_TCP_2_4.*t2.*t5.*t6.*t9.*t10.*t13,0.0,0.0,0.0,1.0,-t2.*t110,t27.*t110,t39+t46+t62+t63+t66+t67+t73+t76+t78+t84+t92+t97+t98+t99+t100+t102+t106+t107-a2.*t3,t27,t2,0.0,-t2.*t109,t27.*t109,t39+t46+t62+t63+t66+t67+t73+t76+t78+t84+t92+t97+t98+t99+t100+t102+t106+t107,t27,t2,0.0,-t2.*t108,t27.*t108,t46+t62+t63+t66+t67+t73+t76+t78+t84+t92+t97+t98+t99+t100+t102+t106+t107,t27,t2,0.0,T_n_TCP_3_4.*t12.*t27+d6.*t8.*t12+T_n_TCP_1_4.*t6.*t7.*t27+T_n_TCP_2_4.*t6.*t13.*t27-d6.*t2.*t3.*t4.*t5.*t6+d6.*t2.*t3.*t6.*t10.*t11+d6.*t2.*t4.*t6.*t9.*t11+d6.*t2.*t5.*t6.*t9.*t10+T_n_TCP_3_4.*t2.*t3.*t4.*t5.*t6-T_n_TCP_3_4.*t2.*t3.*t6.*t10.*t11-T_n_TCP_3_4.*t2.*t4.*t6.*t9.*t11-T_n_TCP_3_4.*t2.*t5.*t6.*t9.*t10-T_n_TCP_1_4.*t2.*t3.*t4.*t5.*t7.*t12+T_n_TCP_1_4.*t2.*t3.*t7.*t10.*t11.*t12+T_n_TCP_1_4.*t2.*t4.*t7.*t9.*t11.*t12+T_n_TCP_1_4.*t2.*t5.*t7.*t9.*t10.*t12-T_n_TCP_2_4.*t2.*t3.*t4.*t5.*t12.*t13+T_n_TCP_2_4.*t2.*t3.*t10.*t11.*t12.*t13+T_n_TCP_2_4.*t2.*t4.*t9.*t11.*t12.*t13+T_n_TCP_2_4.*t2.*t5.*t9.*t10.*t12.*t13,T_n_TCP_3_4.*t2.*t12-d6.*t2.*t12+T_n_TCP_1_4.*t2.*t6.*t7+T_n_TCP_2_4.*t2.*t6.*t13+d6.*t3.*t6.*t8.*t10.*t11+d6.*t4.*t6.*t8.*t9.*t11+d6.*t5.*t6.*t8.*t9.*t10+d6.*t3.*t4.*t5.*t6.*t27+T_n_TCP_3_4.*t3.*t4.*t5.*t6.*t8+T_n_TCP_3_4.*t3.*t6.*t10.*t11.*t27+T_n_TCP_3_4.*t4.*t6.*t9.*t11.*t27+T_n_TCP_3_4.*t5.*t6.*t9.*t10.*t27+T_n_TCP_1_4.*t3.*t7.*t8.*t10.*t11.*t12+T_n_TCP_1_4.*t4.*t7.*t8.*t9.*t11.*t12+T_n_TCP_1_4.*t5.*t7.*t8.*t9.*t10.*t12+T_n_TCP_1_4.*t3.*t4.*t5.*t7.*t12.*t27+T_n_TCP_2_4.*t3.*t8.*t10.*t11.*t12.*t13+T_n_TCP_2_4.*t4.*t8.*t9.*t11.*t12.*t13+T_n_TCP_2_4.*t5.*t8.*t9.*t10.*t12.*t13+T_n_TCP_2_4.*t3.*t4.*t5.*t12.*t13.*t27,t26.*(-T_n_TCP_3_4.*t6+d6.*t6+T_n_TCP_1_4.*t7.*t12+T_n_TCP_2_4.*t12.*t13),sin(t23)./2.0+sin(t40)./2.0,cos(t23).*(-1.0./2.0)+cos(t40)./2.0,t22,T_n_TCP_1_4.*t8.*t12.*t13+T_n_TCP_2_4.*t7.*t12.*t27-T_n_TCP_1_4.*t2.*t3.*t4.*t7.*t11-T_n_TCP_1_4.*t2.*t3.*t5.*t7.*t10-T_n_TCP_1_4.*t2.*t4.*t5.*t7.*t9+T_n_TCP_1_4.*t2.*t7.*t9.*t10.*t11-T_n_TCP_2_4.*t2.*t3.*t4.*t11.*t13-T_n_TCP_2_4.*t2.*t3.*t5.*t10.*t13-T_n_TCP_2_4.*t2.*t4.*t5.*t9.*t13+T_n_TCP_2_4.*t2.*t9.*t10.*t11.*t13-T_n_TCP_1_4.*t2.*t3.*t4.*t5.*t6.*t13+T_n_TCP_2_4.*t2.*t3.*t4.*t5.*t6.*t7+T_n_TCP_1_4.*t2.*t3.*t6.*t10.*t11.*t13+T_n_TCP_1_4.*t2.*t4.*t6.*t9.*t11.*t13+T_n_TCP_1_4.*t2.*t5.*t6.*t9.*t10.*t13-T_n_TCP_2_4.*t2.*t3.*t6.*t7.*t10.*t11-T_n_TCP_2_4.*t2.*t4.*t6.*t7.*t9.*t11-T_n_TCP_2_4.*t2.*t5.*t6.*t7.*t9.*t10,-T_n_TCP_1_4.*t2.*t12.*t13+T_n_TCP_2_4.*t2.*t7.*t12+T_n_TCP_1_4.*t7.*t8.*t9.*t10.*t11+T_n_TCP_1_4.*t3.*t4.*t7.*t11.*t27+T_n_TCP_1_4.*t3.*t5.*t7.*t10.*t27+T_n_TCP_1_4.*t4.*t5.*t7.*t9.*t27+T_n_TCP_2_4.*t8.*t9.*t10.*t11.*t13+T_n_TCP_2_4.*t3.*t4.*t11.*t13.*t27+T_n_TCP_2_4.*t3.*t5.*t10.*t13.*t27+T_n_TCP_2_4.*t4.*t5.*t9.*t13.*t27+T_n_TCP_2_4.*t3.*t4.*t5.*t6.*t7.*t8+T_n_TCP_1_4.*t3.*t6.*t8.*t10.*t11.*t13+T_n_TCP_1_4.*t4.*t6.*t8.*t9.*t11.*t13+T_n_TCP_1_4.*t5.*t6.*t8.*t9.*t10.*t13+T_n_TCP_1_4.*t3.*t4.*t5.*t6.*t13.*t27+T_n_TCP_2_4.*t3.*t6.*t7.*t10.*t11.*t27+T_n_TCP_2_4.*t4.*t6.*t7.*t9.*t11.*t27+T_n_TCP_2_4.*t5.*t6.*t7.*t9.*t10.*t27,-T_n_TCP_1_4.*t3.*t4.*t5.*t7+T_n_TCP_1_4.*t3.*t7.*t10.*t11+T_n_TCP_1_4.*t4.*t7.*t9.*t11+T_n_TCP_1_4.*t5.*t7.*t9.*t10-T_n_TCP_2_4.*t3.*t4.*t5.*t13+T_n_TCP_2_4.*t3.*t10.*t11.*t13+T_n_TCP_2_4.*t4.*t9.*t11.*t13+T_n_TCP_2_4.*t5.*t9.*t10.*t13+T_n_TCP_1_4.*t3.*t4.*t6.*t11.*t13+T_n_TCP_1_4.*t3.*t5.*t6.*t10.*t13+T_n_TCP_1_4.*t4.*t5.*t6.*t9.*t13-T_n_TCP_2_4.*t3.*t4.*t6.*t7.*t11-T_n_TCP_2_4.*t3.*t5.*t6.*t7.*t10-T_n_TCP_2_4.*t4.*t5.*t6.*t7.*t9-T_n_TCP_1_4.*t6.*t9.*t10.*t11.*t13+T_n_TCP_2_4.*t6.*t7.*t9.*t10.*t11,t6.*t27-t2.*t3.*t4.*t5.*t12+t2.*t3.*t10.*t11.*t12+t2.*t4.*t9.*t11.*t12+t2.*t5.*t9.*t10.*t12,t2.*t6+t3.*t8.*t10.*t11.*t12+t4.*t8.*t9.*t11.*t12+t5.*t8.*t9.*t10.*t12+t3.*t4.*t5.*t12.*t27,t30.*(-1.0./2.0)+t44./2.0],[6,6]);
