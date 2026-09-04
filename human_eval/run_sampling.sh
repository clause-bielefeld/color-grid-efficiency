# sample ids + human sample
echo ""
echo "sample ids + human sample"
echo "====================================="
echo "====================================="
echo "====================================="
python make_sample_ids.py 

# model samples
echo ""
echo "model samples"
echo "====================================="
echo "====================================="
echo "====================================="
python make_samples_from_model_descriptions.py --verbose_errors

# make images
echo ""
echo "make images"
echo "====================================="
echo "====================================="
echo "====================================="
python prepare_image_data.py