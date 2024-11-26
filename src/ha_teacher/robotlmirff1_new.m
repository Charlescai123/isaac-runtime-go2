clear;
clc;


T = 1/10;  %sampling period

C = [1, 0, 0];


%%%%Contiunous matrices%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aA = [zeros(1,1),  zeros(1,3),           C,    zeros(1,3);
      zeros(3,1),  zeros(3,3),    zeros(3,3),        eye(3);
      zeros(3,1),  zeros(3,3),    zeros(3,3),    zeros(3,3);
      zeros(3,1),  zeros(3,3),    zeros(3,3),    zeros(3,3)];
aB = [zeros(4,6);
         eye(6)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Discrete-time%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = [zeros(4,6);
         eye(6)]*T;
A = eye(10) + T*aA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%Safety constriant%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b1 = 1/0.1;     % yaw
b2 = 1/0.13;     % height
b3 = 1/0.6;      % velocity
b4 = 1/0.08;      % velocity (yaw rate)


alpha = 1;
n = 12;
D = [ 0,    0,    0,    b1,     0,    0,    0,    0,    0,    0;
    b2,    0,    0,     0,     0,    0,    0,    0,    0,    0;
     0,    0,    0,     0,    b3,    0,    0,    0,    0,    0;
     0,    0,    0,     0,     0,    0,    0,    0,    0,    b4];
% 
% D = [ 0,    0,    0,    b1,     0,    0,    0,    0,    0,    0;
%      b2,    0,    0,     0,     0,    0,    0,    0,    0,    0;
%       0,    0,    0,     0,    b3,    0,    0,    0,    0,    0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta = 0.8;
kappa = 0.1;

beta1 = 0.002;


setlmis([]) 
Q = lmivar(1,[10 1]); 
R = lmivar(2,[6 10]); 

lmiterm([-1 1 1 Q],1,(beta - 2*kappa)*eye(10));
lmiterm([-1 2 1 Q],A,1);
lmiterm([-1 2 1 R],B,1);
lmiterm([-1 2 2 Q],1,0.5);

lmiterm([2 1 1 Q], D, D');
lmiterm([2 1 1 0], -eye(4));

%lmiterm([-3 1 1 Q],1,1);
%lmiterm([-3 2 1 R],1,1);
%lmiterm([-3 2 2 0],(1/beta1)*eye(6));

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);
Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);

P = inv(Q);

aF = round(aB*R*P,0)
eig(aA + aF);

writematrix(P, "P.txt");

M = A + B*R*P;
eig(M'*P*M - P)


