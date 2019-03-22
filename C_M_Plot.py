def matplot_build(value):
	import matplotlib.pyplot as plt 
	import numpy as np
	#%matplotlib inline
	    #<<%Matplotlib will not work in normal Sublime environment
	plt.imshow(value, interpolation='nearest', cmap=plt.cm.Wistia)
	classNames = ['False','True']
	plt.title('Confusion Matrix')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	tick_marks = np.arange(len(classNames))
	plt.xticks(tick_marks, classNames, rotation=45)
	plt.yticks(tick_marks, classNames)

	s = [['TN','FP'], ['FN', 'TP']]

	for i in range(2):
		for j in range(2):
			plt.text(j,i, str(s[i][j])+'='+str(value[i][j]))
	plt.show()			


