import mileage_predict
import plot

"""
Basic logic of the programme:
  - We read the theta in the theta.txt file (initialised at 0 for both theta0 and theta1)
  - We read the data.csv file thanks to the `data_reader()` function
  - We run the model_trainer over 1000 iterations so that the gradient descent may converge.
    *Note:* in this function, we scale the data because it helps with the convergence
    **tremendously**. This explains the `mean`, `std`, `data_reverse`, and `data_scale` functions.
  - Once this is done, we "unscale" the theta, we save it to the theta.txt file, and we draw
    the corresponding plot.
"""

#
# Functions to read the intial parameters
#

def theta_reader(file_name = 'theta.txt'):
    text_file = open(file_name, 'r')
    parameters = {}

    for line in text_file:
        line = line.split('=')
        line[1].replace('\n', '') # Remove line skip
        line[1] = float(line[1]) # Otherwise, value has class <str>
        parameters[line[0]] = line[1]

    text_file.close()
    return parameters


def data_reader(file_name = 'data.csv'):
    # First index corresponds to the mileage; second index to the price
    csv_file = open(file_name, 'r')
    data = []

    for line in csv_file:
        line = line.split(',')
        try:
            line[0], line[1] = float(line[0]), float(line[1])
            data.append(line)
        except ValueError: # If one of the line contains text, we ignore that line
            continue

    csv_file.close()

    return data

#
# Functions to scale and unscale the data
#

def mean(number_list):
    list_mean = sum(number_list)/len(number_list)
    return list_mean

def std(number_list):
    std_error = (mean([number ** 2 for number in number_list]) - mean(number_list) ** 2) ** 0.5
    return std_error

def data_reverse(data):
    return [[line[0] for line in data], [line[1] for line in data]]

def data_scale(data):
    data_column = [[line[0] for line in data], [line[1] for line in data]]
    data_column = [[(number - mean(column)) / std(column) for number in column] for column in data_column]
    new_data = [[line0, line1] for line0, line1 in zip(data_column[0], data_column[1])]
    return new_data


#
# Main function: train the gradient descent model
#

def model_trainer(theta0, theta1, data, learning_rate = 0.1):

    tmp_theta0, tmp_theta1 = 0, 0
    scaled_data = data_scale(data)

    for i in range(0, len(data)):
        mileage = scaled_data[i][0]
        price = scaled_data[i][1]
        estimate_price = mileage_predict.linear_prediction(mileage, theta0, theta1)

        tmp_theta0 += (learning_rate * 1 / len(data)) * (estimate_price - price)
        tmp_theta1 += (learning_rate * 1 / len(data)) * (estimate_price - price) * mileage

    else:
        return {'theta0': theta0 - tmp_theta0, 'theta1': theta1 - tmp_theta1}

#
# Modify the theta to fit the initial, unscaled data, and save it to theta.txt
#

def theta_unscale(theta0, theta1, data):

    reversed_data = data_reverse(data)
    mean0, std0 = mean(reversed_data[0]), std(reversed_data[0])
    mean1, std1 = mean(reversed_data[1]), std(reversed_data[1])

    new_theta0 = std1 * (theta0 - (theta1 * mean0 / std0)) + mean1
    new_theta1 = (theta1) * std1 / std0

    return {'theta0': new_theta0, 'theta1': new_theta1}


def theta_writer(theta0, theta1, file_name = 'theta.txt'):

    text_file = open(file_name, 'w')
    text_file.write('theta0=%f\ntheta1=%f' %(theta0, theta1))
    text_file.close()

#
# Show the MSE
#

def mse_calculator(theta0, theta1, data):

    residuals = [line[1] - mileage_predict.linear_prediction(line[0], theta0, theta1) for line in data]
    mean_squared_error = sum([residual ** 2 for residual in residuals]) / len(residuals)
    return mean_squared_error



if __name__ == '__main__':

    theta = theta_reader()
    data = data_reader()

    for count in range(0, 1000): # A lower number of iterations might work, but why change it?
        theta = model_trainer(theta['theta0'], theta['theta1'], data)
        if count % 100 == 0:
            print('Count: %i' % count)
            print('Theta0: %f    –    Theta1: %f' %(theta['theta0'], theta['theta1']))

    theta = theta_unscale(theta['theta0'], theta['theta1'], data)
    theta_writer(theta['theta0'], theta['theta1'])
    print('Mean Square Error: %f' % mse_calculator(theta['theta0'], theta['theta1'], data))

    plot.line_plot()
