# MLOps-Zoomcamp-2024-Solutions

## MLOps-Zoomcamp-M6-Solution

### Installing pytest

```sh
pipenv install scikit-learn==1.5.0 pandas pyarrow s3fs
pipenv install --dev pytest

mkdir tests
touch tests/__init__.py
touch tests/test_batch.py
```
### Mocking S3 with Localstack

```sh
docker-compose up --build

docker exec -it <id>

aws --version
aws configure list

## Create a new profile for Localstack in your AWS CLI configuration:
aws configure --profile localstack
## Provide any values for the AWS Access Key ID and AWS Secret Access Key
## since Localstack does not validate these credentials.
aws configure set aws_access_key_id test
aws configure set aws_secret_access_key test
aws configure set region us-east-1
aws configure set s3.endpoint_url http://localhost:4566

## Creating a Bucket in Localstack
aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

## Checking Bucket Creation
aws --endpoint-url=http://localhost:4566 s3 ls
aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/

## Upload the Input File to Localstack S3
curl -o yellow_tripdata_2023-03.parquet "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
aws --endpoint-url=http://localhost:4566 s3 cp \
    "./data/yellow_tripdata_2023-03.parquet" \
    "s3://nyc-duration/in/yellow_tripdata_2023-03.parquet"

aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/
```

## Finish the integration test
```sh
!pipenv run python "integraton-test/integration_test.py"

# 2024-07-11 02:14:28,954 [INFO]: Found credentials in environment variables.
# File saved to s3://nyc-duration/in/yellow_tripdata_2023-01.parquet
# 2023 1
# 2024-07-11 02:14:30,175 [INFO]: input_file : s3://nyc-duration/in/yellow_tripdata_2023-01.parquet
# 2024-07-11 02:14:30,175 [INFO]: output_file: s3://nyc-duration/out/yellow_tripdata_2023-01.parquet
# 2024-07-11 02:14:31,686 [INFO]: Found credentials in environment variables.
# 2024-07-11 02:14:32,057 [INFO]: shape : (2, 5)
# predicted mean duration: 18.14
# Sum of predicted durations: 36.27725045203073
```