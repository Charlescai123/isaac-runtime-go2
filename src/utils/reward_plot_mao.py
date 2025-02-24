import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from mpl_toolkits import mplot3d
import scipy.stats as st

L = 400

# s1 = np.loadtxt("1t0.txt", dtype=np.float32)
# s2 = np.loadtxt("1t1.txt", dtype=np.float32)
# s3 = np.loadtxt("1t2.txt", dtype=np.float32)
s4 = np.loadtxt("1t3.txt", dtype=np.float32)
s5 = np.loadtxt("1t4.txt", dtype=np.float32)
# s6 = np.loadtxt("1t5.txt", dtype=np.float32)
# s7 = np.loadtxt("1t6.txt", dtype=np.float32)
s8 = np.loadtxt("1t7.txt", dtype=np.float32)
s9 = np.loadtxt("1t8.txt", dtype=np.float32)
s0 = np.loadtxt("1t9.txt", dtype=np.float32)

# s1 = s1[0 : L]
# s2 = s2[0 : L]
# s3 = s3[0 : L]
s4 = s4[0: L]
s5 = s5[0: L]
# s6 = s6[0 : L]
# s7 = s7[0 : L]
s8 = s8[0: L]
s9 = s9[0: L]
s0 = s0[0: L]

# u1 = np.loadtxt("1u0.txt", dtype=np.float32)
# u2 = np.loadtxt("1u1.txt", dtype=np.float32)
# u3 = np.loadtxt("1u2.txt", dtype=np.float32)
u4 = np.loadtxt("1u3.txt", dtype=np.float32)
u5 = np.loadtxt("1u4.txt", dtype=np.float32)
# u6 = np.loadtxt("1u5.txt", dtype=np.float32)
# u7 = np.loadtxt("1u6.txt", dtype=np.float32)
u8 = np.loadtxt("1u7.txt", dtype=np.float32)
u9 = np.loadtxt("1u8.txt", dtype=np.float32)
u0 = np.loadtxt("1u9.txt", dtype=np.float32)

# u1 = u1[0 : L]
# u2 = u2[0 : L]
# u3 = u3[0 : L]
u4 = u4[0: L]
u5 = u5[0: L]
# u6 = u6[0 : L]
# u7 = u7[0 : L]
u8 = u8[0: L]
u9 = u9[0: L]
u0 = u0[0: L]

ho = 200;

# sX1 = np.convolve(s1, np.ones(ho), 'valid') / ho
# sX2 = np.convolve(s2, np.ones(ho), 'valid') / ho
# sX3 = np.convolve(s3, np.ones(ho), 'valid') / ho
sX4 = np.convolve(s4, np.ones(ho), 'valid') / ho
sX5 = np.convolve(s5, np.ones(ho), 'valid') / ho
# sX6 = np.convolve(s6, np.ones(ho), 'valid') / ho
# sX7 = np.convolve(s7, np.ones(ho), 'valid') / ho
sX8 = np.convolve(s8, np.ones(ho), 'valid') / ho
sX9 = np.convolve(s9, np.ones(ho), 'valid') / ho
sX0 = np.convolve(s0, np.ones(ho), 'valid') / ho

# su1 = np.convolve(u1, np.ones(ho), 'valid') / ho
# su2 = np.convolve(u2, np.ones(ho), 'valid') / ho
# su3 = np.convolve(u3, np.ones(ho), 'valid') / ho
su4 = np.convolve(u4, np.ones(ho), 'valid') / ho
su5 = np.convolve(u5, np.ones(ho), 'valid') / ho
# su6 = np.convolve(u6, np.ones(ho), 'valid') / ho
# su7 = np.convolve(u7, np.ones(ho), 'valid') / ho
su8 = np.convolve(u8, np.ones(ho), 'valid') / ho
su9 = np.convolve(u9, np.ones(ho), 'valid') / ho
su0 = np.convolve(u0, np.ones(ho), 'valid') / ho

# sX10001 = np.convolve(s10001, np.ones(ho), 'valid') / ho
# sX10002 = np.convolve(s10002, np.ones(ho), 'valid') / ho
# sX10003 = np.convolve(s10003, np.ones(ho), 'valid') / ho
# sX10004 = np.convolve(s10004, np.ones(ho), 'valid') / ho
# sX10005 = np.convolve(s10005, np.ones(ho), 'valid') / ho

# TL1 = np.vstack([sX1, sX2, sX3, sX4, sX5, sX6, sX7, sX8, sX9, sX0])
# TL2 = np.vstack([su1, su2, su3, su4, su5, su6, su7, su8, su9, su0])

TL1 = np.vstack([sX4, sX5, sX8, sX9, sX0])
TL2 = np.vstack([su4, su5, su8, su9, su0])
# TL2 = np.vstack([s5001, s5002, s5003, s5004, s5005])
# TL3 = np.vstack([s7001, s7002, s7003, s7004, s7005])
# TL4 = np.vstack([s10001, s10002, s10003, s10004, s10005])
# TL5 = np.vstack([s20001, s20002, s20003])

etCI1 = st.t.interval(confidence=0.95, df=2, loc=TL1.mean(0), scale=st.sem(TL1))
etCI2 = st.t.interval(confidence=0.95, df=2, loc=TL2.mean(0), scale=st.sem(TL2))
# etCI3 = st.t.interval(confidence=0.95, df=2, loc=TL3.mean(0), scale=st.sem(TL3))
# etCI4 = st.t.interval(confidence=0.95, df=2, loc=TL4.mean(0), scale=st.sem(TL4))
# etCI5 = st.t.interval(confidence=0.95, df=2, loc=TL5.mean(0), scale=st.sem(TL5))

etLo1 = np.array(etCI1)[0, :]
etUp1 = np.array(etCI1)[1, :]
etMe1 = TL1.mean(0)

etLo2 = np.array(etCI2)[0, :]
etUp2 = np.array(etCI2)[1, :]
etMe2 = TL2.mean(0)

# etLo3 = np.array(etCI3)[0,:]
# etUp3 = np.array(etCI3)[1,:]
# etMe3 = TL3.mean(0)

# etLo4 = np.array(etCI4)[0,:]
# etUp4 = np.array(etCI4)[1,:]
# etMe4 = TL4.mean(0)

# etLo5 = np.array(etCI5)[0,:]
# etUp5 = np.array(etCI5)[1,:]
# etMe5 = TL5.mean(0)


fig = plt.figure(figsize=(18, 11))
t = np.arange(len(etMe1)) + 1

plt.plot(t, etMe2, linewidth=3, label="Without Unsafe-learning Correction", color='blue')
plt.fill_between(t, etLo2, etUp2, color='blue', alpha=0.2)

plt.plot(t, etMe1, linewidth=3, label="With Unsafe-learning Correction", color='red')
plt.fill_between(t, etLo1, etUp1, color='red', alpha=0.2)

# plt.plot(t, etMe5, linewidth=3, label = "Episode Length: 2000", color='red')
# plt.fill_between(t, etLo5, etUp5, color='red', alpha=0.2,)

# sh1 = 0*np.ones(len(t))
# plt.plot(t,sh1, linewidth=4, color='black',linestyle='dashed',label = "Mission Goal")

plt.legend(ncol=1, loc='upper center', )
plt.rc('legend', fontsize=30)

# plt.xlim(0, 20.5)
plt.ylabel("Learning Reward", fontsize=30)
plt.xlabel("Iteration Steps", fontsize=30)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.grid()
fig.savefig('t1.pdf', dpi=300)
