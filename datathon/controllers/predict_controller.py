from pathlib import Path
# import tensorflow as tf

class PredictController:
    def __init__(self):
        model_path = Path(__file__).parent.parent.parent.joinpath('modelo_dis_v1.keras')
        # self.model = tf.keras.models.load_model(model_path, safe_mode=False)


    def predict(self, user_id):
        shuffle_buffer_size = 1000
        window_size = 5
        batch_size = 32

        dataset = self.get_dataset()

        try:
            return "Hello World"
            # result = self.model_forecast(self.model, dataset, window_size, batch_size)

            # final_result = result.squeeze()

            # cast_result = float(final_result)

            # return f"Resultado previsto: {round(cast_result, 2)}"
        except Exception as e:
            print("Error predicting")
            print(e)
            return None

    def model_forecast(self, model, dataset, window_size, batch_size):
        shuffle_buffer_size = 1000

        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x: x.batch(batch_size))
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(lambda x: (x[:, 0], x[:, 1]))
        return dataset

    def get_dataset(self):
        try:
            # dataset = self.predict_model.get_dataset()
            # symbol = 'DIS'
            # df2 = yf.download(symbol)
            # df_tail = df2.tail(5)

            # forecast_series2 = df_tail['Close']

            # return forecast_series2
            return "Hello World"
        except Exception as e:
            print(e)
            return None

