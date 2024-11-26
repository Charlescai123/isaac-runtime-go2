clear;
clc;


T = 1/10;  %sampling period

%%%%Contiunous matrices%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aA = [zeros(3,3),  zeros(3,3),    1*eye(3),    zeros(3,3);
      zeros(3,3),  zeros(3,3),    zeros(3),    1*eye(3,3);
      zeros(3,3),  zeros(3,3),    zeros(3),    zeros(3,3);
      zeros(3,3),  zeros(3,3),    zeros(3),    zeros(3,3)];
aB = [zeros(6,6);
         eye(6)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Discrete-time%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = [zeros(6,6);
         eye(6)]*T;
A = eye(12) + T*aA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%Safety constriant%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b1 = 1/0.05;     % yaw
b2 = 1/0.13;    % height
b3 = 1/0.6;     % velocity (vx)
% b4 = 1/0.5;     % yaw rate
% b5 = 1/0.5;     % vy

% #  ddq_kp: [ 0.1, 0.1, 100., 100., 100., 0.1 ]
% #  ddq_kd: [ 40., 30., 10., 10., 10., 30. ]

alpha = 1;
n = 12;
D = [0,    0,     0,    0,    0,    b1,     0,    0,    0,    0,    0,    0;
     0,    0,    b2,    0,    0,     0,     0,    0,    0,    0,    0,    0;
     0,    0,     0,    0,    0,     0,    b3,    0,    0,    0,    0,    0;];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


beta = 0.9;
% beta = 0.99;
kappa = 0.05;

% beta1 = 0.2;
beta1 = 0.001;

setlmis([]) 
Q = lmivar(1,[12 1]); 
R = lmivar(2,[6 12]); 

lmiterm([-1 1 1 Q],1,(beta - 2*kappa)*eye(12));
lmiterm([-1 2 1 Q],A,1);
lmiterm([-1 2 1 R],B,1);
lmiterm([-1 2 2 Q],1,0.5);

lmiterm([2 1 1 Q], D, D');
lmiterm([2 1 1 0], -eye(3));

lmiterm([-3 1 1 Q],1,1);
lmiterm([-3 2 1 R],1,1);
lmiterm([-3 2 2 0],(1/beta1)*eye(6));

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);

assert(tmin < 0);
Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);


P = inv(Q);

aF = round(aB*R*P,0)
eig(aA + aF);

writematrix(P, "P.txt");

M = A + B*R*P;
eig(M'*P*M - P)


