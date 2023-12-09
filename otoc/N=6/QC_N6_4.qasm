OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[2],q[1];
h q[2];
rz(dt*0.05136780182793282) q[1];
cx q[2],q[0];
rz(dt*0.02903057544853443) q[0];
cx q[1],q[0];
rz(dt*0.08333443909091853) q[0];
cx q[1],q[0];
cx q[2],q[0];
h q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
h q[1];
s q[2];
h q[2];
cx q[2],q[0];
s q[2];
h q[2];
rz(dt*-0.03180945596099166) q[2];
cx q[1],q[0];
rz(dt*0.05521980237165429) q[0];
cx q[2],q[0];
rz(dt*0.06339689389255035) q[0];
cx q[2],q[0];
cx q[1],q[0];
h q[2];
sdg q[2];
cx q[2],q[0];
h q[2];
sdg q[2];
h q[1];
barrier q[0],q[1],q[2];
s q[0];
h q[0];
h q[2];
cx q[2],q[0];
h q[2];
rz(dt*-0.10745467425961704) q[2];
rz(dt*0.024366007374697834) q[0];
cx q[2],q[0];
rz(dt*0.06102707557568694) q[0];
cx q[2],q[0];
h q[2];
cx q[2],q[0];
h q[2];
h q[0];
sdg q[0];
barrier q[0],q[1],q[2];
cx q[1],q[0];
h q[1];
cx q[2],q[1];
rz(dt*0.10105733579403306) q[1];
rz(dt*-0.07826551045369533) q[0];
cx q[1],q[0];
rz(dt*0.045218677314353285) q[0];
cx q[1],q[0];
cx q[2],q[1];
h q[1];
cx q[1],q[0];
barrier q[0],q[1],q[2];
s q[1];
h q[1];
h q[2];
cx q[2],q[0];
h q[2];
cx q[2],q[1];
rz(dt*0.06990238252518469) q[1];
cx q[2],q[0];
rz(dt*0.000778775532548728) q[0];
cx q[1],q[0];
rz(dt*-0.11036525194336165) q[0];
cx q[1],q[0];
cx q[2],q[0];
cx q[2],q[1];
h q[2];
cx q[2],q[0];
h q[2];
h q[1];
sdg q[1];
barrier q[0],q[1],q[2];
