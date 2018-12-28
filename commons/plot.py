import matplotlib.pyplot as plt
import os
PATH_RES_101_NO_DROPOUT_LOG = "final_result/res_101_nodropout/res_food_loss.log"
PATH_RES_101_NO_DROPOUT_ACC = "final_result/res_101_nodropout/res_food_accuracy.log"
PATH_RES_101_NO_DROPOUT_TEST = "final_result/res_101_nodropout/test_result.log"

PATH_RES_101_DROPOUT_LOG = "final_result/res_101_dropout/res_food_loss.log"
PATH_RES_101_DROPOUT_ACC = "final_result/res_101_dropout/res_food_accuracy.log"
PATH_RES_101_DROPOUT_TEST = "final_result/res_101_dropout/test_result.log"

PATH_RES_50_LOG = "final_result/res_50/res50_food_loss.log"
PATH_RES_50_ACC = "final_result/res_50/res50_food_accuracy.log"
PATH_RES_50_TEST = "final_result/res_50/test_result.log"

PATH_VGG_LOG = "final_result/vgg_food/vgg_food_loss.log"
PATH_VGG_ACC = "final_result/vgg_food/vgg_food_accuracy.log"
PATH_VGG_TEST = "final_result/vgg_food/test_result.log"

PATH_SE_101_DROPOUT_LOG = "final_result/se_101_dropout/se_food_loss.log"
PATH_SE_101_DROPOUT_ACC = "final_result/se_101_dropout/se_food_accuracy.log"
PATH_SE_101_DROPOUT_TEST = "final_result/se_101_dropout/test_result.log"

PATH_SE_101_NO_DROPOUT_LOG = "final_result/se_101_nodropout/se_food_loss.log"
PATH_SE_101_NO_DROPOUT_ACC = "final_result/se_101_nodropout/se_food_accuracy.log"
PATH_SE_101_NO_DROPOUT_TEST = "final_result/se_101_nodropout/test_result.log"

PATH_SE_34_LOG = "final_result/se_34/se34_food_loss.log"
PATH_SE_34_ACC = "final_result/se_34/se34_food_accuracy.log"
PATH_SE_34_TEST = "final_result/se_34/test_result.log"

PATH_SE_18_LOG = "final_result/se_18/se18_food_loss.log"
PATH_SE_18_ACC = "final_result/se_18/se18_food_accuracy.log"
PATH_SE_18_TEST = "final_result/se_18/test_result.log"

PATH_SE_50_LOG = "final_result/se_50/se50_food_loss.log"
PATH_SE_50_ACC = "final_result/se_50/se50_food_accuracy.log"
PATH_SE_50_TEST = "final_result/se_50/test_result.log"

OUTPUT_LOSS_OVER_INTER = "loss_over_iter.png"
OUTPUT_AVGLOSS_OVER_INTER = "avgloss_over_iter.png"
OUTPUT_ACC_OVER_EPOCS = "[IMG]Training Accuracy"
OUTPUT_AVG_ACC = "[IMG]Average Accuracy For Models"
OUTPUT_RES_SE_101_VGG = "[IMG]Res101- Se101 - vgg"
OUTPUT_RES_SE_50_VGG = "[IMG]se50 - res50 - vgg"
OUTPUT_RES_SE_101_NODROPOUT = "[IMG]res101 - se 101 - nodropout"
OUTPUT_ALL = "[IMG]All AVG LOSS OVER ITER"

class Plot_G():
	def __init__(self):
		pass
	def draw_training_acc(self):
		X1, Y1 = self.read_acc_log(PATH_RES_101_DROPOUT_ACC) #epochs and acc
		X2, Y2 = self.read_acc_log(PATH_RES_101_NO_DROPOUT_ACC) #epochs and acc
		X3, Y3 = self.read_acc_log(PATH_RES_50_ACC) #epochs and acc
		X4, Y4 = self.read_acc_log(PATH_VGG_ACC) #epochs and acc
		X5, Y5 = self.read_acc_log(PATH_SE_101_DROPOUT_ACC) #epochs and acc
		X6, Y6 = self.read_acc_log(PATH_SE_101_NO_DROPOUT_ACC) #epochs and acc
		X7, Y7 = self.read_acc_log(PATH_SE_34_ACC) #epochs and acc
		X8, Y8 = self.read_acc_log(PATH_SE_50_ACC) #epochs and acc
		X9, Y9 = self.read_acc_log(PATH_SE_18_ACC)
		plt.plot(X1, Y1, label = 'RES_101_DROPOUT')
		plt.plot(X2, Y2, label = 'RES_101_NO_DROPOUT')
		plt.plot(X3, Y3, label = 'RES_50')
		plt.plot(X4, Y4, label = 'VGG_16')
		plt.plot(X5, Y5, label = 'SE_101_DROPOUT')
		plt.plot(X6, Y6, label = 'SE_101_NO_DROPOUT')
		plt.plot(X7, Y7, label = 'SE_34')
		plt.plot(X8, Y8, label = 'SE_50')
		plt.plot(X9, Y9, label = 'SE_18')
		plt.xlabel('Number of Epochs')
		plt.ylabel('Accuracy')
		plt.title('Training Accuracy Over Epochs')
		plt.legend()
		plt.savefig(OUTPUT_ACC_OVER_EPOCS)
		plt.close()


	def draw_loss_over_iter(self):
		# fig = plt.figure()
		X1, Y1, Z1 = self.read_log(PATH_RES_101_DROPOUT_LOG) #batch num, loss, avg_loss
		X2, Y2, Z2 = self.read_log(PATH_RES_101_NO_DROPOUT_LOG)
		X3, Y3, Z3 = self.read_log(PATH_RES_50_LOG)
		X4, Y4, Z4 = self.read_log(PATH_VGG_LOG)
		X5, Y5, Z5 = self.read_log(PATH_SE_101_DROPOUT_LOG)
		X6, Y6, Z6 = self.read_log(PATH_SE_101_NO_DROPOUT_LOG)
		X7, Y7, Z7 = self.read_log(PATH_SE_34_LOG)
		X8, Y8, Z8 = self.read_log(PATH_SE_50_LOG)
		X9, Y9, Z9 = self.read_log(PATH_SE_18_LOG)

		# Res101- Se101 - vgg
		plt.plot(X1, Z1, label = 'RES_101_DROPOUT')
		plt.plot(X5, Z5, label = 'SE_101_DROPOUT')
		plt.plot(X4, Z4, label = 'VGG_16')
		plt.legend()
		plt.xlabel('Batch number')
		plt.ylabel('Loss')
		plt.title('Loss Over Iteration Graph')
		plt.savefig(OUTPUT_RES_SE_101_VGG)
		plt.close()
		# se50 - res50 - vgg
		plt.plot(X8, Z8, label = 'SE_50')
		plt.plot(X3, Z3, label = 'RES_50')
		plt.plot(X4, Z4, label = 'VGG_16')
		plt.legend()
		plt.xlabel('Batch number')
		plt.ylabel('Loss')
		plt.title('Loss Over Iteration Graph')
		plt.savefig(OUTPUT_RES_SE_50_VGG)
		plt.close()
		#se101 - se101-ko dropout
		plt.plot(X2, Z2, label = 'RES_101_NO_DROPOUT')
		plt.plot(X6, Z6, label = 'SE_101_NO_DROPOUT')
		plt.legend()
		plt.xlabel('Batch number')
		plt.ylabel('Loss')
		plt.title('Loss Over Iteration Graph')
		plt.savefig(OUTPUT_RES_SE_101_NODROPOUT)
		plt.close()
		#all avg_loss of all models
		plt.plot(X1, Z1, label = 'RES_101_DROPOUT')
		plt.plot(X2, Z2, label = 'RES_101_NO_DROPOUT')
		plt.plot(X3, Z3, label = 'RES_50')
		plt.plot(X4, Z4, label = 'VGG_16')
		plt.plot(X5, Z5, label = 'SE_101_DROPOUT')
		plt.plot(X6, Z6, label = 'SE_101_NO_DROPOUT')
		plt.plot(X7, Z7, label = 'SE_34')
		plt.plot(X8, Z8, label = 'SE_50')
		plt.plot(X9, Z9, label = 'SE_18')
		plt.legend()
		plt.xlabel('Batch number')
		plt.ylabel('Loss')
		plt.title('Loss Over Iteration Graph')
		plt.savefig(OUTPUT_ALL)
		plt.close()

		pass

	def draw_test_acc(self):
		X1, Y1, Z1 = self.read_test_log(PATH_RES_101_DROPOUT_TEST) #X list classes, Y list acc for classes, Z avg acc of that model
		X2, Y2, Z2 = self.read_test_log(PATH_RES_101_NO_DROPOUT_TEST)
		X3, Y3, Z3 = self.read_test_log(PATH_RES_50_TEST)
		X4, Y4, Z4 = self.read_test_log(PATH_VGG_TEST)
		X5, Y5, Z5 = self.read_test_log(PATH_SE_101_DROPOUT_TEST)
		X6, Y6, Z6 = self.read_test_log(PATH_SE_101_NO_DROPOUT_TEST)
		X7, Y7, Z7 = self.read_test_log(PATH_SE_34_TEST)
		X8, Y8, Z8 = self.read_test_log(PATH_SE_50_TEST)
		X9, Y9, Z9 = self.read_test_log(PATH_SE_18_TEST)
		model_names = ["RE101_DO", "RE101_NO_DO", "RES_50", "VGG_16", "SE101_DO", "SE101_NO_DO",
		 "SE_34", "SE_50", "SE_18"]
		avg_acc = [Z1, Z1, Z3, Z4, Z5, Z6, Z7, Z8, Z9]
		classes_name = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
		plt.barh(model_names, avg_acc, alpha = 1)
		plt.title('Average Accuracy Of Models')
		plt.xlabel('Models')
		plt.ylabel('Avg Accuracy')
		plt.yticks(fontsize = 6.5)
		plt.savefig(OUTPUT_AVG_ACC)
		plt.close()

		#draw test for classes in each model
		self.draw_test_for_1_model(classes_name, Y1, title = 'RES_101_DROPOUT TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y2, title = 'RES_101_NO_DROPOUT TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y3, title = 'RES_50 TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y4, title = 'VGG_16 TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y5, title = 'SE_101_DROPOUT TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y6, title = 'SE_101_NO_DROPOUT TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y7, title = 'SE_34 TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y8, title = 'SE_50 TEST RESULT')
		self.draw_test_for_1_model(classes_name, Y9, title = 'SE_18 TEST RESULT')		
		pass


	def draw_test_for_1_model(self, classes, acc_list, title): #draw test for classes in 1 model
		# print ("X: {} lenght: {}".format(classes, len(classes)))
		# print ("Y: {} lenght: {}".format(acc_list, len(acc_list)))
		plt.bar(classes, acc_list, alpha = 1)
		plt.title(title)
		plt.xlabel('Classes')
		plt.ylabel('Accuracy')
		plt.savefig("[IMG]"+title)
		plt.close()




	def read_test_log(self, path_test_log):
		X = [] #classes
		Y = [] #acc for class
		count = 0
		file = open(path_test_log,"r")
		for line in file:
			if(count <= 10):
				X.append(count)
			list_of_nums = line.split(":")
			# print(count)
			# print("log {}".format(list_of_nums[1].split(" ")[1].split(" ")[0].split("%")[0]))
			if(count <= 10):
				Y.append(float(list_of_nums[1].split(" ")[1].split(" ")[0].split("%")[0]))
			else:
				Z = float(list_of_nums[1].split(" ")[1].split(" ")[0].split("%")[0]) #average accuracy
			count = count + 1
		return X, Y, Z #X classes, Y acc, Z avg acc
		pass

	def read_acc_log(self, path_log):
		X = [] #epochs
		Y = [] #accuracy
		epoch = 0
		file = open(path_log,"r")
		for line in file:
			X.append(epoch)
			list_of_nums = line.split(":")
			Y.append(float(list_of_nums[1].split(" ")[1]))
			epoch = epoch + 1
		return X, Y
		pass

	def read_log(self, path_log):
		batch_size = 0
		file = open(path_log,"r")
		X = []
		Y = []
		Z = []
		for line in file:
			list_of_nums = line.split(" ")
			# print(list_of_nums)
			# epoc_no = int(list_of_nums[0].split(',')[0].split('[')[1])
			batch_size+=1
			batch_no = int(list_of_nums[1].split(']')[0])
			true_batch_no = batch_size
			# print("epoc: {}, batch: {}".format(epoc_no, batch_no))
			loss = float(list_of_nums[3])
			avg_loss = float(list_of_nums[8])
			# print("loss {} avg_loss {}".format(loss, avg_loss))
			X.append(true_batch_no)
			Y.append(loss)
			Z.append(avg_loss)
		return X, Y, Z
		pass
	# folder = "../"
	# file= os.path.join(folder,"train", "log" ,"res_food.log")

	# file = open(PATH,"r")
	# print('name {}'.format(file.name))

	# x = [2, 4, 6]
	# y = [1, 3, 5]
	# plt.plot(x, y)
	# plt.show()

