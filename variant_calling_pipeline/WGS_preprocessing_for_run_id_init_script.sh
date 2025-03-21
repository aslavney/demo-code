#! /bin/bash

set -euo pipefail

# automatically terminate instance after 40 hours
# note: this does not seem to be working for r5.4xlarge isntances as of March 2019
(sleep 144000; bash ${SRC}/ec2_terminate_this_instance.sh) &

# Expected environment variables:
: "${SRA_RUN_ID:?SRA_RUN_ID must be set}"
: "${S3_PROJECT_DIRECTORY:?S3_PROJECT_DIRECTORY must be set}"
: "${S3_OUTPUT_DIRECTORY:?S3_OUTPUT_DIRECTORY must be set}"
: "${OUTPUT_ENVIRONMENT:?OUTPUT_ENVIRONMENT must be set}"
: "${SRC:?SRC must be set}"  # set when instance launched

echo "SRA_RUN_ID: ${SRA_RUN_ID}"
echo "OUTPUT_ENVIRONMENT: ${OUTPUT_ENVIRONMENT}"
echo "S3_PROJECT_DIRECTORY: ${S3_PROJECT_DIRECTORY}"
echo "S3_OUTPUT_DIRECTORY: ${S3_OUTPUT_DIRECTORY}"

source ${SRC}/paths.sh

job_started_message="{\"task\": \"started\", \"sra_run_id\": \"${SRA_RUN_ID}\"}"
job_started_queue_url=${sqs_root}/${QUEUE_NAME_PREFIX}-started
retry aws sqs send-message --queue-url ${job_started_queue_url} --message-body "${job_started_message}"

# Create and cd into a working directory, so that the preprocessing script can just work in its current directory
mkdir -p ${HOME}/data
cd ${HOME}/data

# Copy additional needed files and software from S3
aws s3 sync s3://${S3_PROJECT_DIRECTORY} ${HOME}/data/WGS_ref_files --only-show-errors

# Activate science conda env (has necessary software)
conda activate science

# Run preprocessing steps
bash run_WGS_preprocessing_for_run_id_in_SQS.sh ${SRA_RUN_ID} ${S3_OUTPUT_DIRECTORY}

timestamp=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
s3_logfile=s3://${S3_OUTPUT_DIRECTORY}/wgs_logs/${SRA_RUN_ID}_preprocessing_${timestamp}.log.txt
echo "Copying logs to S3: ${s3_logfile}"
retry s4cmd --force put ${HOME}/WGS_preprocessing_for_run_id_init_script.sh.log.txt ${s3_logfile}

job_done_message="{\"task\": \"done\", \"sra_run_id\": \"${SRA_RUN_ID}\"}"
job_done_queue_url=${sqs_root}/${QUEUE_NAME_PREFIX}-done
retry aws sqs send-message --queue-url ${job_done_queue_url} --message-body "${job_done_message}"

echo "FINISHED"

# terminate instance after complete
sleep 10
bash ${SRC}/ec2_terminate_this_instance.sh

