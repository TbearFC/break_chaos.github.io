# 使用tf serving部署模型

## 先保存成可用的模型

    saver = tf.train.Saver()
    saver.save(sess,
                   'checkpoint_directory/linear_100_epochs.ckpt',
                   global_step=epoch)
    builder = tf.saved_model.builder.SavedModelBuilder("./model_complex")
    signature = saved_model.predict_signature_def(inputs={'Input': X},
                                      outputs={'Output': y_model})
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[saved_model.tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()


## 模型的restore

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./model_complex")
        graph = tf.get_default_graph()

        x = sess.graph.get_tensor_by_name('Input:0')
        y = sess.graph.get_tensor_by_name('Output:0')
        scores = sess.run(y,
                          feed_dict={x: x_train})
        print(scores,y_train)

## 开启serving

使用tf serving docker版本

docker run -t --rm -p 8501:8501 \
    -v /Users/bear/PycharmProjects/DLpipeline/model_complex:/models/model \
    -e MODEL_NAME=model \
    tensorflow/serving &
    