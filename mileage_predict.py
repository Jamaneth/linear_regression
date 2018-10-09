import mileage_train

def linear_prediction(mileage, theta0, theta1):

    return theta0 + theta1 * mileage

if __name__ == '__main__':

    theta = mileage_train.theta_reader()
    user_mileage = float(input('Please enter a mileage: '))

    print(linear_prediction(user_mileage, theta['theta0'], theta['theta1']))
