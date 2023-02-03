# run on lightning cloud (fully managed)
lightning run app app.py --cloud

# run on a cluster you created called pikachu
lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc
lightning run app app.py --cloud pikachu

# run on a cluster you created called bolt
lightning create cluster bolt --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc
lightning run app app.py --cloud bolt
