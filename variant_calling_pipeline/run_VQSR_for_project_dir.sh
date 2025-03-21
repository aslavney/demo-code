#! /bin/bash

set -euo pipefail

# Run GATK VQSR on raw SNPs for all datasets in s3_project_dir
# This script assumes that you are using a high memory instance (e.g. r5.4xlarge) with at least 500GB of storage and all software installed

s3_project_dir=$1
mkdir ./data
cd ./data

# Copy additional needed files and software from S3
aws s3 sync s3://${s3_project_dir} ${HOME}/data/WGS_ref_files --only-show-errors
DATA_DIR=${HOME}/data/WGS_ref_files

# Copy the vcf.gz and index files for all samples in s3_project_dir
aws s3 cp s3://${s3_project_dir}/BQSR_vcfs/ ./ --recursive --exclude "*" --include "*_all.raw.vcf.gz*" --only-show-errors

# Combine the vcf.gz files from all samples into one vcf
touch input_variant_files.list
ls *_all.raw.vcf.gz > input_variant_files.list
bcftools merge --file-list input_variant_files.list --regions chr1,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr2,chr20,chr21,chr22,chr23,chr24,chr25,chr26,chr27,chr28,chr29,chr3,chr30,chr31,chr32,chr33,chr34,chr35,chr36,chr37,chr38,chr4,chr5,chr6,chr7,chr8,chr9,chrX,chrM,Dog_Y --threads 7 --output-type v --output ${s3_project_dir}_all.raw.vcf

# Create separate files for the SNPs and indels
echo "Separating SNPs, indels"
gatk SelectVariants --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -V ${s3_project_dir}_all.raw.vcf --select-type SNP -O ${s3_project_dir}_all.raw.snps.vcf
# we don't have a good reference dataset for indels, so not trying to call them for now
#gatk SelectVariants --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -V ${s3_project_dir}_all.raw.vcf --select-type INDEL -O ${s3_project_dir}_all.raw.indels.vcf

# Calculate recalibrated quality scores for SNPs based on training set
echo "Running VariantRecalibrator for SNPs `date`"
gatk VariantRecalibrator -R ${DATA_DIR}/canFam3.1_and_SRY.fa -V ${s3_project_dir}_all.raw.snps.vcf --resource illumina,known=false,training=true,truth=true,prior=12.0:${DATA_DIR}/illumina.sites_w_id.cf3.1.vcf -an DP -an QD -an FS -an MQRankSum -an FS -an ReadPosRankSum -an MQ --max-gaussians 4 -mode SNP -O ${s3_project_dir}.recal.snps.recal --tranches-file ${s3_project_dir}.recal.snps.tranches --rscript-file ${s3_project_dir}.recal.snps.plots.R

# Get final list of high-confidence SNPs by using recalibrated scores for variant calling
echo "Applying VQSR to raw SNPs in $run_id data `date`"
gatk ApplyVQSR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -V ${s3_project_dir}_all.raw.snps.vcf --truth-sensitivity-filter-level 99.0 -tranches-file ${s3_project_dir}.recal.snps.tranches -recal-file ${s3_project_dir}.recal.snps.recal -mode SNP -O ${s3_project_dir}.recal.snps.vcf

# Convert recal vcf to gvcf and index (so we can quickly pull regions of interest with bcftools view)
bcftools view ${s3_project_dir}.recal.snps.vcf -Oz -o ${s3_project_dir}.recal.snps.vcf.gz
bcftools index ${s3_project_dir}.recal.snps.vcf.gz

# Copy all recal SNP files to s3
aws s3 cp ./ s3://${s3_project_dir}/QC_VQSR_vcfs/ --recursive --exclude "*" --include "*.recal.snps.*"

echo "Finished all steps for $s3_project_dir `date`"




