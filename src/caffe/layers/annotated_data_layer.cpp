#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  heatmap_c_ = anno_data_param.heatmap_c();
  heatmap_h_ = anno_data_param.heatmap_h();
  heatmap_w_ = anno_data_param.heatmap_w();
  visualise_ = anno_data_param.heatmap_visual();
  sigma_ = anno_data_param.heatmap_sigma();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
	if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
	  int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
		LOG(INFO) << "num_bboxes: " << num_bboxes;
	  }
	  
	}
    vector<int> label_shape(4, 1);
	label_shape[0] = batch_size;
	label_shape[1] = heatmap_c_;
	label_shape[2] = heatmap_h_;
	label_shape[3] = heatmap_w_;
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
	int tmpnum = 0;
	for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
        tmpnum += anno_datum.annotation_group(g).annotation_size();
    }
	//LOG(INFO) << "tmpnum in anno_datum : " << tmpnum;
    //read_time += timer.MicroSeconds();
    //timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);

	
    //timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
		int tmpnum = 0;
		for (int g = 0; g < sampled_datum->annotation_group_size(); ++g) {
			tmpnum += sampled_datum->annotation_group(g).annotation_size();
		}
		
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
		  //LOG(INFO) << "num_bboxes in data transformer: " << num_bboxes;
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    //trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
	const int label_channel_size = heatmap_h_ * heatmap_w_;
    const int label_img_size = label_channel_size * heatmap_c_;
	float sigma = sigma_;
	cv::Mat dataMatrix = cv::Mat::zeros(heatmap_h_, heatmap_w_, CV_32FC1);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX && heatmap_c_ == 1) {
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
		
        for (int item_id = 0; item_id < batch_size; ++item_id) {
			
			for (int top_p = item_id*label_img_size; top_p < (item_id + 1)*label_img_size; top_p++)
				top_label[top_p] = 0;
			
			const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
			
			for (int g = 0; g < anno_vec.size(); ++g) {
				const AnnotationGroup& anno_group = anno_vec[g];
				for (int a = 0; a < anno_group.annotation_size(); ++a) {
					const Annotation& anno = anno_group.annotation(a);
					const NormalizedBBox& bbox = anno.bbox();
					int px_ = bbox.xmin()*heatmap_w_;
					int py_ = bbox.ymin()*heatmap_h_;
					//LOG(INFO) << "key point :  " << px_ << "," << py_;
					for (int i = 0; i < heatmap_h_; i++) {
						for (int j = 0; j < heatmap_w_; j++) {
							int label_idx = item_id * label_img_size + i * heatmap_w_ + j;
							float gaussian = 0.5 * ( pow(i - py_, 2.0) + pow(j - px_, 2.0) ) * pow(1 / sigma, 2.0);
							if (gaussian > 4.6052) 
								continue;
							gaussian = exp(-gaussian);

							top_label[label_idx] = top_label[label_idx] > gaussian ? top_label[label_idx] : gaussian;
							
							if (top_label[label_idx] > 1)
								top_label[label_idx] = 1;
							
						}
                    }
                }
            }
			
			//LOG(INFO) << "batch size =  " << item_id ;
			
			if (visualise_) {
				for (int i = 0; i < heatmap_h_; i++) {
					for (int j = 0; j < heatmap_w_; j++) {
						int label_idx = item_id * label_img_size + i * heatmap_w_ + j;
						dataMatrix.at<float>((int)i, (int)j) = top_label[label_idx];
					}
				}
			}
				
			if (visualise_)
				cv::imshow("guass image", dataMatrix);
			if (visualise_)
				cv::waitKey(0);
        }
    } else if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX && heatmap_c_ > 1) {
		top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
		
        for (int item_id = 0; item_id < batch_size; ++item_id) {
			
			for (int top_p = item_id*label_img_size; top_p < (item_id + 1)*label_img_size; top_p++)
				top_label[top_p] = 0;
			const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
			for (int g = 0; g < anno_vec.size(); ++g) {   // --- attention: type [g] - point label
				const AnnotationGroup& anno_group = anno_vec[g];
				int ptype_channel_idx = g;
				for (int a = 0; a < anno_group.annotation_size(); ++a) {
					const Annotation& anno = anno_group.annotation(a);
					const NormalizedBBox& bbox = anno.bbox();
					int px_ = bbox.xmin()*heatmap_w_;
					int py_ = bbox.ymin()*heatmap_h_;
					//LOG(INFO) << "key point :  " << px_ << "," << py_;
					for (int i = 0; i < heatmap_h_; i++) {
						for (int j = 0; j < heatmap_w_; j++) {
							int label_idx = item_id * label_img_size + g*label_channel_size + i * heatmap_w_ + j;
							float gaussian = 0.5 * ( pow(i - py_, 2.0) + pow(j - px_, 2.0) ) * pow(1 / sigma, 2.0);
							if (gaussian > 4.6052) 
								continue;
							gaussian = exp(-gaussian);

							top_label[label_idx] = top_label[label_idx] > gaussian ? top_label[label_idx] : gaussian;
							
							if (top_label[label_idx] > 1)
								top_label[label_idx] = 1;
							
						}
                    }
                }
            }
			
			//LOG(INFO) << "batch size =  " << item_id ;
			
			if (visualise_) {
				for (int c = 0; c < heatmap_c_; c++) {
					for (int i = 0; i < heatmap_h_; i++) {
						for (int j = 0; j < heatmap_w_; j++) {
							int label_idx = item_id * label_img_size + c * label_channel_size + i * heatmap_w_ + j;
							dataMatrix.at<float>((int)i, (int)j) = top_label[label_idx];
						}
					}
					cv::imshow("guass image", dataMatrix);
					cv::waitKey(0);
				}
			}
		}
	}
	
  }
  //timer.Stop();
  //batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
