
//-----------------    Variables and Parameters   -------------------------

var A C Y K N R W K_N theta;
varexo eA eTheta;

parameters beta sigma phi alpha delta rhoA sigmaA rhoTheta sigmaTheta;


// Parameters
	
beta = 0.96;
sigma = 3;
phi = 4;
alpha = 0.33;
delta = 0.075;
rhoA = 0.45;
sigmaA = 0.005;
rhoTheta = 0.6;
sigmaTheta = 0.01;


//-------------------------     Model     ---------------------------------

model;
// Production
Y = A * K(-1)^alpha * N^(1 - alpha);

// Household Constraint
C + K - (1 - delta) * K(-1) = W * N + R * K(-1);

// Euler's Equation
1 = beta * ((C(+1) - theta(+1) * N(+1)^(1 + phi) / (1 + phi)) / 
    (C - theta * N^(1 + phi) / (1 + phi)))^(-sigma) * (1 + R(+1) - delta);

// Labor 
W = theta * N^phi;

// Firm's Maximisation Problem
R = alpha * A * (K(-1)/N)^(alpha - 1);
W = (1 - alpha) * A * (K(-1)/N)^alpha;

// K_N Ratio
K_N = K(-1)/N;

// Dynamics of Shocks
log(A) = rhoA * log(A(-1)) + eA;
log(theta) = rhoTheta * log(theta(-1)) - eTheta; // - eTheta for negative
                                                 //  realization 

end;


//-----------------------    Steady State     -----------------------------

initval;

A = 1;             
theta = 1;          

R = 1/beta - 1 + delta;

K_N = ((alpha * A) / R)^(1/(1 - alpha));

W = (1 - alpha) * A * K_N^alpha;

N = (W / theta)^(1/phi);

K = K_N * N;

Y = K^alpha * N^(1 - alpha);

C = W * N + (R - delta) * K;

end;

steady;
resid;


//--------------------  Specification of Shocks  --------------------------

shocks;

var eA; stderr sigmaA;                             
var eTheta; stderr sigmaTheta;

end;


//-------------------------  Computation  ---------------------------------

stoch_simul(irf=50) A C Y K N R W K_N theta;
