import matplotlib.pyplot as plt

#figure
fig = plt.figure( )
fig2 = plt.figure( figsize =
plt.figaspect(2,0) )

#axes
fig.add_axes( )
ax1 = fig.add_subplot( 221 )
#row-col-num
ax3 = fig.add_subplot( 212 )
fig3, axes = plt.subplots(
nrows=2, ncols=2)
fig4, axes2 = plt.subplots( ncols=3 )
