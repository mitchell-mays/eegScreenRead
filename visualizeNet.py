import webbrowser, os

class Visualizer:
	def __init__(self, data, labels):
		self.DATA = data
		self.LABELS = labels
		self
		
	def showVisualization(self, visualizationDirectory="NN_Visualize/"):
		dataString = ""
		for i in range(len(self.DATA)):
			dataRowString = ','.join(map(str,self.DATA[i]))
			dataRowString += "|"
			dataRowString += ','.join(map(str,self.LABELS[i]))
			dataString += dataRowString
			
			if (i < (len(self.DATA)-1)):
				dataString += "\r"
			
		visDataFile = open(visualizationDirectory+"data.txt", 'w')
		visDataFile.write(dataString)
		visDataFile.close()
		
		ie = webbrowser.get(webbrowser.iexplore)
		ie.open('file://' + os.path.realpath(visualizationDirectory+'index.html'))
			
		