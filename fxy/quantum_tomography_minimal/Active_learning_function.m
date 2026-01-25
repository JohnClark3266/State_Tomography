%% Active_learning_function

%% Define operators

dim = 30;

sigmax = [0, 1; 1, 0];
sigmay = [0, -1i; 1i, 0];
sigmaplus = 0.5 * (sigmax - 1i * sigmay);
sigmadown = 0.5 * (sigmax + 1i * sigmay);
sigmaz = sigmaplus * sigmadown;

sz = sparse(tensor(sigmaz, qeye(dim)));
sx = sparse(tensor(sigmax, qeye(dim)));
sy = sparse(tensor(sigmay, qeye(dim)));
splus = sparse(tensor(sigmaplus, qeye(dim)));
sdown = sparse(tensor(sigmadown, qeye(dim)));

adag = tensor(qeye(2), create(dim));
a = tensor(qeye(2), destroy(dim));

g = basis(0);
e = basis(1);

K = 2 * pi * 5.3e-6;
chi = 2 * pi * 1.93e-3;

H0 = - K/2 * adag * (adag * a) * a - chi * adag * a * sz;

T1 = 555e3;
T2 = 678e3;
kappa_1 = 1/T1;
kappa_phi_total = 1/T2 - 1/T1/2;
nth = 0.04;
T1q = 135e3;
T2q = 135e3;
kappa_2 = 1/T1q;
kappa_phi2 = 1/T2q - 1/T1q/2;
nth2 = 0.04;

kappa_phi_additional = kappa_2/2 * real(sqrt((1+1i*chi/kappa_2)^2+...
    4i*chi*nth2/kappa_2)-1);

kappa_phi_pure = kappa_phi_total - kappa_phi_additional;

cops = {};
cops{end+1} = sqrt((nth + 1) * kappa_1) * a;
cops{end+1} = sqrt(nth * kappa_1) * adag;
cops{end+1} = sqrt(2 * kappa_phi_pure) * adag * a;
cops{end+1} = sqrt((nth2 + 1) * kappa_2) * sdown;
cops{end+1} = sqrt(nth2 * kappa_2) * splus;
cops{end+1} = sqrt(2 * kappa_phi2) * splus * sdown;
eops = {};
% cops = {};

%% benchmarking

alpha_list = py_matrix_coords;

state = 1;

switch state
    case 1
        psi_target = fock(dim, 3);
        rho_target = tensor(g, psi_target);
    case 2
        psi_target = coherent(dim, 2i);
        rho_target = tensor(g, psi_target);
    case 3
        dT = 1000;
        psi0 = coherent(dim, 2i);
        rho0 = tensor(g, psi0);
        rho_target = QId(dT, rho0, H0, cops, 1);
        psi_target = ptrace(rho_target, [1], [2, dim]);
    case 4
        psi_target = unit(coherent(dim, 2i) + coherent(dim, -2i));
        rho_target = tensor(g, psi_target);
end

n = size(py_matrix_state, 2);

for i = 1: n
    if py_matrix_state(1, i) == 1
        beta = py_matrix_coords(i);
        py_matrix_wigner(i) = computeWigner_real(beta, rho_target, H0, a, adag, sdown, splus, sz, cops, dim);
        py_matrix_state(i) = 2;
    end
end

xvec = linspace(-5, 5, 64);
yvec = linspace(-5, 5, 64);
Wigner_target = wignerFunction(psi_target,xvec,yvec,2);

mat_matrix_state = py_matrix_state;
mat_matrix_wigner = py_matrix_wigner;

function W = computeWigner_real(alpha, rho, H0, a, adag, sdown, splus, sz, cops, dim)

mean_delta1 = 0;
std_delta1 = 30e-9;

mean_delta2 = 0;
std_delta2 = 5e-6;

delta1 = mean_delta1 + std_delta1 * randn;
delta2 = mean_delta2 + std_delta2 * randn;

Hd = H0 + delta1 * adag * a + delta2 * sz;

if size(rho, 1) == size(rho, 2)
    psi1 = rho;
else
    psi1 = rho * rho';
end

width1 = 20;
gap_OP_OP = 10;
coeff = 1;
width2 = 24 * coeff;
Omega = pi / 4 / width2;
Tpioverchi = 228.5134;

alpha_fluctuated = alpha * (1 + 0.01 * randn);
Omega_fluctuated = Omega * (1 + 0.01 * randn);
Omega_fluctuated1 = Omega * (1 + 0.01 * randn);
Omega_fluctuated2 = Omega * (1 + 0.01 * randn);

psi2 = Displacement_onlyC(-alpha_fluctuated, width1, psi1, Hd, cops, a, adag, 0);
psi3 = QId(gap_OP_OP, psi2, Hd, cops, 1);
psi4 = Selective(Omega_fluctuated, width2, psi3, Hd, cops, sdown, splus, adag, a, 0, 'qubit');
psi5 = QId(Tpioverchi, psi4, Hd, cops, 1);
psi6 = Selective(Omega_fluctuated1, width2, psi5, Hd, cops, sdown, splus, adag, a, 0, 'qubit');
psi7 = Selective(Omega_fluctuated2, width2, psi5, Hd, cops, sdown, splus, adag, a, pi, 'qubit');

qubit_state1 = ptrace(psi6, [2], [2, dim]);
qubit_state2 = ptrace(psi7, [2], [2, dim]);

F1 = cal_fidelity(qubit_state1, basis(0));
F2 = cal_fidelity(qubit_state2, basis(0));

W = (2 / pi) * (F2 - F1);

end

function output = Displacement_onlyC(alpha, width, psi, H0, cops, a, adag, theta)

tlist = linspace(0,width,2);
Hd = 1.i * (alpha * adag * exp(1i * theta) - alpha' * a * exp(-1i * theta))/width;
results = mesolve(H0+Hd, {}, tlist, psi, cops,{});
output = (results.state{end});
end

function output = Selective(Omega, width, psi, H0, cops, sdown, splus, adag, a, theta, system_type)

if strcmpi(system_type, 'qubit')
    Hd = 1i * Omega * sdown * exp(1i * theta) - 1i *  Omega' * splus * exp(-1i * theta);
    tlist = linspace(0, width, 2);
    results = mesolve(H0 + Hd, {}, tlist, psi, cops, {});
    output = (results.state{end});

elseif strcmpi(system_type, 'cavity')
    Hd = Omega * adag * exp(1i * theta) + Omega' * a * exp(-1i * theta);
    tlist = linspace(0, width, 2);
    results = mesolve(H0 + Hd, {}, tlist, psi, cops, {});
    output = (results.state{end});

else
    error('Invalid system type. Please choose either "qubit" or "cavity".');
end

end

function fidelity = cal_fidelity(rou_Meas, rou_ideal)
if size(rou_Meas,1) ~= size(rou_Meas,2)
    rou_Meas = rou_Meas*rou_Meas';
end
if size(rou_ideal,1) ~= size(rou_ideal,2)
    rou_ideal = rou_ideal*rou_ideal';
end
% fidelity = abs(trace(sqrtm(sqrtm(rou_ideal)*rou_Meas*sqrtm(rou_ideal))))^2;
fidelity = abs(trace(rou_Meas'*rou_ideal));

end

function output=QId(width, psi, H0, cops, dt)
    tlist = linspace(0, width * dt, 2);
    results = mesolve(H0, {}, tlist, psi, cops, {});
    output = (results.state{end});
end