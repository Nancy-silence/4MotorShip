plt.subplot(1,3,1)
plt.plot(t,x,label="x")
plt.plot(t,y,label="y")
plt.legend()
plt.subplot(1,3,2)
plt.plot(t,x_v,label="x")
plt.plot(t,y_v,label="y")
plt.plot(t,v_all,label="all")
plt.legend()
plt.subplot(1,3,3)
plt.plot(t,x_a,label="x")
plt.plot(t,y_a,label="y")
plt.legend()