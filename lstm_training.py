##Performing 50 iterations of model training to get the highest accuracy and reduced loss 
# epochs is number of iterations performed in model training.
N_epochs = 50
batch_size = 1024

saver = tf.train.Saver()
history = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_count = len(X_train)

for i in range(1, N_epochs + 1):
	for start, end in zip(range(0, train_count, batch_size), 
						range(batch_size, train_count + 1, batch_size)):
		sess.run(optimizer, feed_dict={X: X_train[start:end],
									Y: Y_train[start:end]})
	_, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
		X: X_train, Y: Y_train})
	_, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
		X: X_test, Y: Y_test})
	history['train_loss'].append(loss_train)
	history['train_acc'].append(acc_train)
	history['test_loss'].append(loss_test)
	history['test_acc'].append(acc_test)

	if (i != 1 and i % 10 != 0):
		print(f'epoch: {i} test_accuracy:{acc_test} loss:{loss_test}')
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], 
											feed_dict={X: X_test, Y: Y_test})
print()
print(f'final results : accuracy : {acc_final} loss : {loss_final}')
	
##Accuracy graph 
plt.figure(figsize=(12,8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r--", label="Test loss")
plt.plot(np.array(history['test_acc']), "g--", label="Test accuracy")

plt.title("Training session's progress over iteration")
plt.legend(loc = 'upper right', shadow = True)
plt.ylabel('Training Progress(Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()
## Confusion matrix 
max_test = np.argmax(Y_test, axis=1)
max_predictions = np.argmax(predictions, axis = 1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix, xticklabels = LABELS, yticklabels = LABELS, annot =True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel('Predicted_label')
plt.ylabel('True Label')
plt.show()
























