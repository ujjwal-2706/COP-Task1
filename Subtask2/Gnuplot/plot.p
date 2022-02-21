set term png size 3840,2160 font 'Helvetica,30' linewidth 2
set output 'times_pthread.png'
set xrange [0:300]
set yrange [0:10000]
set boxwidth 1
plot 'threadnew.dat' using 1:3:2:6:5 with candlesticks lt -1 lw 2 whiskerbars,\
    '' using 1:4:4:4:4 with candlesticks lt -1 lw 2 notitle
