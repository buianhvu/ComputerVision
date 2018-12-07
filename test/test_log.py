import os

PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "log")
default_file1 = "default.log"
default_file2 = "default_unknown.log"


def write_log(classes, class_correct, class_total, accuracy: float,path=PATH, file_name=default_file1):
    num_classes = len(classes)
    save_file = os.path.join(path, file_name)
    f = open(save_file, "w")
    for i in range(num_classes):
        f.write("Accuracy of %5s: %2d%%\n"%(classes[i], 100 * class_correct[i]/class_total[i]))
    f.write("Accuracy: %f\n"%(accuracy*100))
    f.close()


def write_log_unknown(class_correct, class_total, accuracy: float, path=PATH, file_name=default_file2):
    save_file = os.path.join(path, file_name)
    f = open(save_file, "w")
    for key in class_total:
        if key not in class_correct:
            f.write("Accuracy of %5s: 0%%\n")
        f.write("Accuracy of %d: %2d%%\n" % (key, 100 * class_correct[key] / class_total[key]))
    f.write("Accuracy: %f\n" % (accuracy*100))
    f.close()