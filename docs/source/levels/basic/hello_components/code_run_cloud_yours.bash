# first create a cluster (creation could take ~30 minutes)
lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc

# run on that cluster
lightning run app app.py --cloud pikachu
