#! /bin/bash

set -euo pipefail

# This script assumes that:
# 1) You are using a high memory instance (e.g. r5.4xlarge) with at least 500GB of storage (assuming each fasta.gz file is ~20GB)
# 2) It is being run by WGS_preprocessing_for_run_id_init_script.sh in an ec2 instance that will auto terminate once this script completes

# run_id = SRA run ID (prefix SRR or ERR, e.g. SRR1784082), s3_project_dir = dir on s3 to store outputs
run_id=$1
s3_project_dir=$2
mkdir ./$run_id
cd ./$run_id

DATA_DIR=${HOME}/data/WGS_ref_files

# Data download and alignment
bash /repo/align_fastq_to_ref_genome.sh run_id s3_project_dir

# Mark duplicate reads (these are a product of sequencing, or PCR amplification itself)
# Note: some picard operations often fail on large files with the default java -Xmx parameter value specified in the bioconda picard.sh file (1g),
# run it via java instead of bioconda picard so you can specify this parameter (the job will fail if the heap memory cap is either too low or too high)
echo "Marking duplicates for $run_id `date`"
java -Xmx104g -jar ${DATA_DIR}/picard.jar MarkDuplicates I=${run_id}.sorted.bam O=${run_id}.dedup.bam M=${run_id}.dedup_metrics CREATE_INDEX=true VERBOSITY=ERROR
rm ${run_id}.sorted.bam*

# Add read group information according to the sequence batches represented in these samples
echo "Adding read groups for $run_id `date`"
java -Xmx104g -jar ${DATA_DIR}/picard.jar AddOrReplaceReadGroups I=${run_id}.dedup.bam O=${run_id}.rg.bam RGID=SRA1 RGLB=libSRA1 RGPL=illumina RGPU=unit1 RGSM=${run_id} SORT_ORDER=coordinate CREATE_INDEX=true VERBOSITY=ERROR
rm ${run_id}.dedup*

## Base recalibration using known polymorphic sites (BQSR)
echo "Generating BQSR table for $run_id `date`"
gatk --java-options "-Xss100m" BaseRecalibratorSpark --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -I ${run_id}.rg.bam --known-sites ${DATA_DIR}/illumina.sites_w_id.cf3.1.vcf -O ${run_id}_bqsr.table

echo "Applying BQSR to $run_id data `date`"
gatk --java-options "-Xss100m" ApplyBQSRSpark --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -I ${run_id}.rg.bam --bqsr-recal-file ${run_id}_bqsr.table -O ${run_id}_bqsr.bam
samtools index -\@ 16 ${run_id}_bqsr.bam

echo "Copying $run_id rg bam files to s3://${s3_project_dir}/raw_sorted_alignments/ (for coverage calculation) `date`"
aws s3 cp ./ s3://${s3_project_dir}/raw_sorted_alignments/ --recursive --exclude "*" --include "*rg.bam*" --only-show-errors
rm ${run_id}.rg*

# Initial raw variant calling (HaplotypeCaller + GenotypeGCVFs) for variants on canonical canFam3.1 autosomes
echo "Running HaplotypeCaller for $run_id `date`"
mkdir ./partial
cd ./partial
parallel --halt now,fail=1 --jobs 7 --verbose "gatk HaplotypeCaller --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -I ../${run_id}_bqsr.bam --emit-ref-confidence GVCF -L chr{} -ploidy 2 -O ${run_id}_{}.diploid.raw.gvcf 2>&1" ::: {1..38}

echo "Generating raw variant vcfs for $run_id `date`"
parallel --halt now,fail=1 --jobs 7 --verbose "gatk GenotypeGVCFs --verbosity ERROR -R ${DATA_DIR}/canFam3.1_and_SRY.fa -V ${run_id}_{}.diploid.raw.gvcf -O ${run_id}_{}.diploid.raw.vcf" ::: {1..38}
rm *.diploid.raw.gvcf*

# Combine all of the chromosome vcf files into one
touch input_variant_files.list
ls *.diploid.raw.vcf > input_variant_files.list
gatk MergeVcfs --VERBOSITY ERROR -I input_variant_files.list -O ../${run_id}_all.raw.vcf
rm *.diploid.raw.vcf*

cd ../

# Convert the variant vcf to vcf.gz and index
bcftools view ${run_id}_all.raw.vcf -Oz -o ${run_id}_all.raw.vcf.gz
bcftools index ${run_id}_all.raw.vcf.gz

echo "Copying $run_id vcf.gz files to s3://${s3_project_dir}/BQSR_vcfs/ `date`"
aws s3 cp ./ s3://${s3_project_dir}/BQSR_vcfs/ --recursive --exclude "*" --include "${run_id}_all.raw*" --only-show-errors

echo "Finished all preprocessing and BQSR steps for $run_id `date`"

