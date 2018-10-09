import matplotlib.pyplot as plt
import mileage_train
import mileage_predict

def line_plot():

    data = mileage_train.data_reader()
    theta = mileage_train.theta_reader()

    reversed_data = mileage_train.data_reverse(data)
    predict_data = [[line[0] for line in data],\
    [mileage_predict.linear_prediction(line[0], theta['theta0'], theta['theta1']) for line in data]]

    plt.plot(reversed_data[0], reversed_data[1], 'ro')
    plt.plot(predict_data[0], predict_data[1])
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.show()

if __name__ == '__main__':

    line_plot()
