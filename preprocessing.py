import pandas as pd
from pandas.api.types import is_string_dtype
import os
import datetime
import numpy as np
import itertools
from sklearn import preprocessing
import glob


class processFile:
	def __init__(self, datatype):
		self.dtype = datatype

class preprocessor:
	def __init__(self, doLog=False, logFileLocation=os.getcwd()+"\preprocessor.log", datafolder="", datafile="", datatype=""):
		self.files = {}
		self.data = []
		self.columnNameValueEquivalents = {}
		self.logFile = logFileLocation
		self.log = doLog
		self.fileClassInfo = {}
		
		if (self.log):
			log = open(self.logFile, 'a')
			log.write("\r\n\r\n\r\n")
			log.write("==========================================================================================\r\n")
			log.write("==========================================================================================\r\n")
			log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "      New Logged Session\r\n")
			log.write("==========================================================================================\r\n")
			log.write("==========================================================================================\r\n")
			log.close()
		
		if (datafile != ""):
			if (not os.path.isfile(datafile)):
				raise Exception("Error -- Datafile does not exist")
			else:
				if (datatype == ""):
					datatype = datafile.split(".")[-1]
				
				self.files[datafile] = processFile(datatype)
				this.load(datafile)
		if (datafolder != ""):
			if (not os.path.isdir(datafolder)):
				raise Exception("Error -- Datafolder does not exist")
			else:
				this.load(datafolder=datafolder)

	def load(self, datafile="", datafolder="", datatype="", useFileNameAsDataPoint=""):
		if (not os.path.isfile(datafile) and not os.path.isdir(datafolder)):
			print("Error -- Datafile or Datafolder does not exist")
		else:
			if (datafile != ""):
				myFiles = [datafile]
			else:
				myFiles = glob.glob(datafolder+"*.*")
				
			#print(myFiles)
			for dfile in myFiles:
				datafile = dfile
				
				
				
				type = datatype
				if (type == ""):
					if (not datafile in self.files):
						type = datafile.split(".")[-1]
						self.files[datafile] = datatype
					else:
						type = self.files[datafile].dtype
				
				fileLoaded = ""
				if (type == "csv"):
					fileLoaded = pd.read_csv(datafile, encoding='utf-8')
				elif (type == "txt"):
					fileLoaded = pd.read_csv(datafile, sep="\t", encoding='utf-8')
				elif (type == "json"):
					fileLoaded = pd.read_json(datafile, encoding='utf-8')
				elif (type == "html"):
					fileLoaded = pd.read_html(datafile, encoding='utf-8')
				elif ((type == "xls") or (type == "xlsx")):
					fileLoaded = pd.read_excel(datafile, encoding='utf-8')
				elif (type == "hdf"):
					fileLoaded = pd.read_hdf(datafile, encoding='utf-8')
				elif (type == "sql"):
					fileLoaded = pd.read_sql(datafile, encoding='utf-8')
				else:
					raise Exception("Unreadable file type...Failing...")
				
				if (useFileNameAsDataPoint != ""):
					fileLoaded[useFileNameAsDataPoint] = datafile.split("\\")[-1].split(".")[0]
				fileLoaded = fileLoaded.replace(u'\xa0', u' ')
				self.data.append(fileLoaded)
	
	#////////////////////////////////////////////////////////
	#////////////////////////////////////////////////////////
	#						Merge
	#////////////////////////////////////////////////////////
	#////////////////////////////////////////////////////////
	#	Description:
	#		Combine any data files in self.files that have matching  
	#		columns. args allow for looser matching...etc
	#
	# 	Structure of equivalents
	#	[
	#		[name1, alsoName1]
	#		[name2, myName2, thirdpossibleName2, 2]
	#	]
	def merge(self, trust=False, requireExactHeaderMatch=True, numHeaderMatchRequired=1, equivalents=[], removeDups=True):
		merging = True
		if (trust):
			if (self.log):
				log = open(self.logFile, 'a')
				log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Forced Trust, merging all files named under first file header\r\n")
				log.close()
			print("Forced Trust, merging all files named under first file header")
		
			#Final Safety Check
			numCols = len(self.data[0].columns)
			for file in self.data:
				if (len(file.columns) != numCols):
					raise Exception("Files do not contain same number of columns...Failing...")
					
			#Merge all files
			done = False
			while(not done):
				main = 0
				adding = len(self.data)-1
				
				#rename all columns in adding
				for col in range(len(self.data[main].columns)):
					self.data[adding] = self.data[adding].rename(columns = {self.data[adding].columns[col]:self.data[main].columns[col]})

				self.data[main] = pd.concat([self.data[main],self.data[adding]], ignore_index=True)
				self.data = self.data[:-1]
				if (len(self.data) == 1):
					done = True
			
			if (self.log):
				log = open(self.logFile, 'a')
				log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "All files successfully merged\r\n")
				log.close()
			print("All files successfully merged")
			
			if (removeDups):
				#self.data[0] = self.data[0].drop_duplicates()
				if (self.log):
					log = open(self.logFile, 'a')
					log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Duplicates Removed\r\n")
					log.close()
				print("Duplicates Removed")
				
				
			
		else:
			while(merging):
				merging = False
				newData = []
				for files in range(len(self.data)):
					for filesSec in range(files+1,len(self.data)):
						hasAllHeaders = True
						sameHeaders = []
						for header in list(self.data[files]):
							if (not header in list(self.data[filesSec])):
								for row in equivalents:
									found = False
									if header in row:
										for equivHeader in row:
											if (equivHeader in list(self.data[filesSec])):
												#This is a poor way of doing things, it will assign the new column name
												#even if it does not end up merging
												self.data[filesSec] = self.data[filesSec].rename(columns = {equivHeader:header})
												found = True
												
								if (not found):
									hasAllHeaders = False
								else:
									sameHeaders.append(header)
							else:
								sameHeaders.append(header)
								
						if ((requireExactHeaderMatch and (not hasAllHeaders)) or ((not requireExactHeaderMatch) and (len(sameHeaders) >= numHeaderMatchRequired))):
							break
						else:
							if (self.log):
								log = open(self.logFile, 'a')
								log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Merging file no: " + files + " with file no: " + filesSec + "\r\n")
								log.close()
							print("Merging file no: " + files + " with file no: " + filesSec)
							
							
							merging = True
							newDataFile = pd.concat([self.data[files],self.data[filesSec]], ignore_index=True)
							if (removeDups):
								newDataFile = newDataFile.drop_duplicates()
							
							newData.append(newDataFile)
							for addRestOfFiles in range(files+1, len(self.data)):
								if (addRestOfFiles != filesSec):
									newData.append(self.data[addRestOfFiles])
									
							break
							
					if (merging):
						break
						
					newData.append(self.data[files])
				self.data = newData
	
	# Here you can write a string to be 'eval'ed for the purpose of creating a new column
	# The df in question is written as 'df' and then replaced inside function
	#
	#		Example:
	#			proc.addColumnFromData("GreaterThan","df[\"Open\"] > df[\"Close\"]") 
	def addColumnFromData(self, newColumnName, formulaString, fileLocation=-1):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			self.data[fileNum][newColumnName] = eval(formulaString.replace("df","self.data[fileNum]"))
			
	
	def showFiles(self, fileLoc=-1):
		if (fileLoc < 0):
			for file in self.data:
				print(file)
		else:
			print(self.data[fileLoc])
	
	#Checks to see if a column has less unique values then threshold. If so it replaces unique
	# values with a number to represent.
	def makeNumeric(self, fileLocation=-1, columns=[], classThreshold=20):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			if (len(columns) > 0):
				#take the following column names and replace unique values with 
				for col in columns:
					temp = self.data[fileNum][col].dropna().unique()
					
					for cell in range(len(temp)):
						if (self.log):
							log = open(self.logFile, 'a')
							log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Replacing File[" + str(fileNum) + "]:" + temp[cell] + " with: " + str(cell-1)  + "\r\n")
							log.close()
						print("Replacing File[" + str(fileNum) + "]:" + temp[cell] + " with: " + str(cell))
						#Cell-1 is in order to be zero based
						self.data[fileNum][col].replace(temp[cell], cell, True)
					
					
			
			#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			#//////////////////////////////////////////
			# If no columns are given it will try and 
			# guess the columns that are classes
			#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			#//////////////////////////////////////////
			else:
				for col in self.data[fileNum].columns:
					temp = self.data[fileNum][col].dropna().unique()
					
					# current logic is: If you have less unique values than half of the rows
					# it is likely a "class" value, not a continuous value
					if (len(temp) < classThreshold):
						if (self.log):
							log = open(self.logFile, 'a')
							log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Deemed " + str(fileNum) + ": " + str(col) + " to be non-continuous" + "\r\n")
							log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Making classes in " + str(fileNum) + ": " + str(col) + " numeric" + "\r\n")
							log.close()
						print("Deemed " + str(fileNum) + ": " + str(col) + " to be non-continuous")
						print("Making classes in " + str(fileNum) + ": " + str(col) + " numeric")
							
						for cell in range(len(temp)):
							if (self.log):
								log = open(self.logFile, 'a')
								log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Replacing File[" + str(fileNum) + "]:" + temp[cell] + " with: " + str(cell)  + "\r\n")
								log.close()
							print("Replacing File[" + str(fileNum) + "]:" + str(temp[cell]) + " with: " + str(cell))
							self.data[fileNum][col].replace(temp[cell], cell, True)
						
	#Takes a column that should be numeric, and removes any values that are non-numeric, changing type to numeric
	def cleanNumeric(self, fileLocation=-1, columns=[]):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
			
		for fileNum in itList:
			if (len(columns)==0):
				columns = self.data[fileNum].columns
				
			self.data[fileNum][columns] = self.data[fileNum][columns].apply(pd.to_numeric, errors='coerce')

	
	#Normalizes all columns, if columns are given then only those
	def normalizeColumns(self, fileLocation=-1, columns=[]):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
			
		for fileNum in itList:
			if (len(columns) <= 0):
				columns = self.data[fileNum].columns
				#take the following column names and replace unique values with 
			for col in columns:
				if (np.issubdtype(self.data[fileNum][col].dtype,np.number)):
					self.data[fileNum][col]=(self.data[fileNum][col]-self.data[fileNum][col].min())/(self.data[fileNum][col].max()-self.data[fileNum][col].min())
				else:
					raise Exception(col + " data type is non-numeric: " + str(self.data[fileNum][col].dtype) + "...Failing...")
			if (self.log):
				log = open(self.logFile, 'a')
				log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Normalized File:" + str(fileNum) + "\r\n")
				log.close()
			print("Normalized File:" + str(fileNum))
				
	#Takes all classes in a columns, assigns them their own columns with 1 or 0 to indicate if that class
	def makeOutput(self, column, fileLocation=-1):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			self.fileClassInfo[fileNum] = column
		
			if (not np.issubdtype(self.data[fileNum][column].dtype, np.number)):
				self.makeNumeric(fileNum, columns=[column])
		
			temp = self.data[fileNum][column].max()
			newDF = self.data[fileNum].drop(column, axis=1)
			for colClass in range(int(temp+1)):
				newDF[column+"_"+str(colClass)] = self.data[fileNum][column].apply(lambda x: int(x==colClass))
			
			#print(newDF)
			self.data[fileNum] = newDF
			
			if (self.log):
				log = open(self.logFile, 'a')
				log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "In File:" + str(fileNum) + " column: " + column + " has been made into labels" +"\r\n")
				log.close()
			print("In File:" + str(fileNum) + " column: " + column + " has been made into labels")
			
	#Separates data into data/labels 
	#			if makeoutput was already called then uses that output
	#			else expects value in column name
	#			    (currently does not handle continuous values)
	def getDataAndLabels(self, column_name="", fileLocation=-1):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		allFilesLabels = []
		allFilesData = []
		for fileNum in itList:
			if column_name != "":
				my_class_col = column_name
			else:
				if fileNum in self.fileClassInfo:
					my_class_col = self.fileClassInfo[fileNum]
				else:
					raise Exception("No class column given, and none save from MakeOutput call")
		
			columnBuild = []
			for col in self.data[fileNum].columns:
				if my_class_col+"_" in col:
					columnBuild.append(col)
					
			allLabels = self.data[fileNum][columnBuild].copy()
			allData = self.data[fileNum].drop(columnBuild, axis=1)
			allFilesLabels.append(allLabels.values)
			allFilesData.append(allData.values)
			
		if len(itList) == 1:
			return allFilesData[0], allFilesLabels[0]
		else:
			return allFilesData, allFilesLabels
	
	#Finds a subset of the data with no non existent values -- according to a few rules
	def getCleanSubset(self, fileLocation=-1, subsetOptions="maxdata", minCols=-1, minRows=-1, forceSaveColumns=[]):
		#Possible subsetOptions values:
		#							maxdata = finds the subset that contains the most data fields in total (rows*columns)
		#							maxrows = prioritizes finding a subset that encompasses the most rows possible, at the cost of losing columns
		#							maxcols = prioritizes finding a subset that encompasses the most columns possible, at the cost of losing rows
		sOptions = ["maxdata","maxrows","maxcols"]
		if (not subsetOptions in sOptions):
			raise Exceptions("subsetOptions incorrect -- possible values are: " + str(sOptions))
		
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			maxVal = 0
			comboVal = []
			if (len(self.data[fileNum]) == len(self.data[fileNum].dropna(axis=0, how='any'))):
				print("Full dataset is clean")
				
			else:
				combinations = []
				for L in range(len(self.data[fileNum].columns)):
					for subset in itertools.combinations(self.data[fileNum].columns, L):
						allFound = True
						for neededCol in forceSaveColumns:
							if (neededCol in subset):
								allFound = False
								break

						if allFound:
							#print(list(subset))
							combinations.append(list(subset))
					
				for combo in combinations:
					temp = self.data[fileNum].drop(combo, axis=1)
					temp = temp.dropna(axis=0, how='any')
					if ((len(temp) > minRows) and (len(temp.columns) > minCols)):
							if subsetOptions == "maxdata":
								if ((len(temp)*len(temp.columns)) > maxVal):
									maxVal = (len(temp)*len(temp.columns))
									comboVal = combo
							elif subsetOptions == "maxrows":
								if (len(temp) > maxVal):
									maxVal = len(temp)
									comboVal = combo
							else:
								if (len(temp.columns) > maxVal):
									maxVal = len(temp.columns)
									comboVal = combo
				
				if (maxVal != 0):
					if (self.log):
						log = open(self.logFile, 'a')
						log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Clean dataset found for File:" + str(fileNum) + "\r\n")
						log.close()
					print("Clean dataset found for File:" + str(fileNum))
					self.data[fileNum] = self.data[fileNum].drop(comboVal, axis=1).dropna(axis=0, how='any')
				else:
					raise Exception("No clean subset found with given rules...Failing...")
	
	
	# I am currently working on this piece -- could be pretty crucial in cleaning up dirty data pieces.
	#		current status:
	#				- Works for simple split with given column and delimeter -- handles expected columns and errLeft
	#				- have not started to attemp smartAssess or regex/not sure what I want to do here
	def splitColumns(self, fileLocation=-1, columns=[], delim="", expectedColumns=2, errLeft=True, regex=False, smartAssess=False):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			if (len(columns) == 0):
				columns = self.data[fileNum].columns
				
			for column in columns:
				if (delim != ""):
					if (errLeft):
						split_df = self.data[fileNum][column].apply(lambda x: pd.Series(x.split(delim)))
					else:
						split_df = self.data[fileNum][column].apply(lambda x: pd.Series([i for i in reversed(x.split(delim))]))
						
					if (len(split_df.columns) > expectedColumns):
						if (self.log):
							log = open(self.logFile, 'a')
							log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Warning -- new Columns exceed expected columns value of: " + str(expectedColumns) + "\r\n")
							log.write('{:%Y-%m-%d %H:%M:%S}>>>> '.format(datetime.datetime.now()) + "Forcing Expected Column Limit" + "\r\n")
							log.close()
						print("Warning -- new Columns exceed expected columns value of: " + str(expectedColumns))
						print("Forcing Expected Column Limit")
						
						split_df = split_df[split_df.columns[0:expectedColumns-len(split_df.columns)]]
							
					
					self.data[fileNum] = self.data[fileNum].drop([column], axis=1)
					if (len(split_df.columns) == 1):
						self.data[fileNum][column] = split_df[split_df.columns[0]]
						if (self.data[fileNum][column].str.isnumeric().all()):
							self.data[fileNum][column] = pd.to_numeric(self.data[fileNum][column])
					else:
						for newCol in range(len(split_df.columns)):							
							self.data[fileNum][column+"_"+str(newCol)] = split_df[split_df.columns[newCol]]
							if (self.data[fileNum][column+"_"+str(newCol)].str.isnumeric().all()):
								self.data[fileNum][column+"_"+str(newCol)] = pd.to_numeric(self.data[fileNum][column+"_"+str(newCol)])
							
							
			
						
	#=======================================================
	#=======================================================
	#					Minor tools
	#=======================================================
	#=======================================================
	
	#Forwards drop command to dataframe
	def drop(self, index, fileLocation=-1, axis=0):
		dropAxis = axis
	
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
		
		for fileNum in itList:
			self.data[fileNum] = self.data[fileNum].drop(index,axis=dropAxis)
	
	#Randomizes data
	def shuffle(self, fileLocation=-1):
		itList = []
		if (fileLocation < 0):
			for i in range(len(self.data)):
				itList.append(i)
		else:
			itList.append(fileLocation)
			
		for fileNum in itList:
			self.data[fileNum] = self.data[fileNum].sample(frac=1).reset_index(drop=True)