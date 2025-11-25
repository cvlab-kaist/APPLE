import numpy as np
import pickle
import tensorflow as tf


def tfLite_weight_to_torch(weight):
    # tfLite (output_channels, filter_height, filter_width, input_channels)
    # torch (output_channels, input_channels, filter_height, filter_width)
    return np.transpose(weight, (0, 3, 1, 2))

if __name__ == "__main__":

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./data/iris_landmark.tflite", experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.load("./data/test.npy")
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data_eyeContour = interpreter.get_tensor(output_details[0]['index'])
    output_data_iris = interpreter.get_tensor(output_details[1]['index'])

    details = interpreter.get_tensor_details()

    # print(details[18:32])

    input = np.array(interpreter.get_tensor(details[0]['index']))
    Conv2D_0_weight = np.array(interpreter.get_tensor(details[1]['index'])) # (output_channels, filter_height, filter_width, input_channels)
    Conv2D_0_bias = np.array(interpreter.get_tensor(details[2]['index']))
    Conv2D_0_output = np.array(interpreter.get_tensor(details[3]['index']))

    Prelu_0_weight = np.array(interpreter.get_tensor(details[4]['index']))
    Prelu_0_output = np.array(interpreter.get_tensor(details[5]['index']))
    Conv2D_block_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[6]['index']))
    Conv2D_block_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[7]['index']))
    Conv2D_block_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[8]['index']))
    Conv2D_block_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[9]['index']))
    Conv2D_block_0_Prelu_0_output = np.array(interpreter.get_tensor(details[10]['index']))
    Conv2D_block_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[11]['index']))
    Conv2D_block_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[12]['index']))
    Conv2D_block_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[13]['index']))
    Conv2D_block_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[14]['index']))
    Conv2D_block_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[15]['index']))
    Conv2D_block_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[16]['index']))
    Conv2D_block_0_output = np.array(interpreter.get_tensor(details[17]['index']))

    Prelu_1_weight = np.array(interpreter.get_tensor(details[18]['index']))
    Prelu_1_output = np.array(interpreter.get_tensor(details[19]['index']))
    Conv2D_block_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[20]['index']))
    Conv2D_block_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[21]['index']))
    Conv2D_block_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[22]['index']))
    Conv2D_block_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[23]['index']))
    Conv2D_block_1_Prelu_0_output = np.array(interpreter.get_tensor(details[24]['index']))
    Conv2D_block_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[25]['index']))
    Conv2D_block_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[26]['index']))
    Conv2D_block_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[27]['index']))
    Conv2D_block_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[28]['index']))
    Conv2D_block_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[29]['index']))
    Conv2D_block_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[30]['index']))
    Conv2D_block_1_output = np.array(interpreter.get_tensor(details[31]['index']))

    Prelu_2_weight = np.array(interpreter.get_tensor(details[32]['index']))
    Prelu_2_output = np.array(interpreter.get_tensor(details[33]['index']))
    Conv2D_block_2_Conv2D_0_weight = np.array(interpreter.get_tensor(details[34]['index']))
    Conv2D_block_2_Conv2D_0_bias = np.array(interpreter.get_tensor(details[35]['index']))
    Conv2D_block_2_Conv2D_0_output = np.array(interpreter.get_tensor(details[36]['index']))
    Conv2D_block_2_Prelu_0_weight = np.array(interpreter.get_tensor(details[37]['index']))
    Conv2D_block_2_Prelu_0_output = np.array(interpreter.get_tensor(details[38]['index']))
    Conv2D_block_2_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[39]['index']))
    Conv2D_block_2_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[40]['index']))
    Conv2D_block_2_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[41]['index']))
    Conv2D_block_2_Conv2D_1_weight = np.array(interpreter.get_tensor(details[42]['index']))
    Conv2D_block_2_Conv2D_1_bias = np.array(interpreter.get_tensor(details[43]['index']))
    Conv2D_block_2_Conv2D_1_output = np.array(interpreter.get_tensor(details[44]['index']))
    Conv2D_block_2_output = np.array(interpreter.get_tensor(details[45]['index']))

    Prelu_3_weight = np.array(interpreter.get_tensor(details[46]['index']))
    Prelu_3_output = np.array(interpreter.get_tensor(details[47]['index']))
    Conv2D_block_3_Conv2D_0_weight = np.array(interpreter.get_tensor(details[48]['index']))
    Conv2D_block_3_Conv2D_0_bias = np.array(interpreter.get_tensor(details[49]['index']))
    Conv2D_block_3_Conv2D_0_output = np.array(interpreter.get_tensor(details[50]['index']))
    Conv2D_block_3_Prelu_0_weight = np.array(interpreter.get_tensor(details[51]['index']))
    Conv2D_block_3_Prelu_0_output = np.array(interpreter.get_tensor(details[52]['index']))
    Conv2D_block_3_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[53]['index']))
    Conv2D_block_3_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[54]['index']))
    Conv2D_block_3_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[55]['index']))
    Conv2D_block_3_Conv2D_1_weight = np.array(interpreter.get_tensor(details[56]['index']))
    Conv2D_block_3_Conv2D_1_bias = np.array(interpreter.get_tensor(details[57]['index']))
    Conv2D_block_3_Conv2D_1_output = np.array(interpreter.get_tensor(details[58]['index']))
    Conv2D_block_3_output = np.array(interpreter.get_tensor(details[59]['index']))

    Prelu_4_weight = np.array(interpreter.get_tensor(details[60]['index']))
    Prelu_4_output = np.array(interpreter.get_tensor(details[61]['index']))
    Conv2D_block_v2_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[62]['index']))
    Conv2D_block_v2_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[63]['index']))
    Conv2D_block_v2_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[64]['index']))
    Conv2D_block_v2_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[65]['index']))
    Conv2D_block_v2_0_Prelu_0_output = np.array(interpreter.get_tensor(details[66]['index']))
    Conv2D_block_v2_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[67]['index']))
    Conv2D_block_v2_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[68]['index']))
    Conv2D_block_v2_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[69]['index']))
    Conv2D_block_v2_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[71]['index']))
    Conv2D_block_v2_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[72]['index']))
    Conv2D_block_v2_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[73]['index']))
    Conv2D_block_v2_0_maxpool_output = np.array(interpreter.get_tensor(details[70]['index']))
    Conv2D_block_v2_0_padding = np.array(interpreter.get_tensor(details[74]['index']))
    Conv2D_block_v2_0_padding_output = np.array(interpreter.get_tensor(details[75]['index']))
    Conv2D_block_v2_0_output = np.array(interpreter.get_tensor(details[76]['index']))

    Prelu_5_weight = np.array(interpreter.get_tensor(details[77]['index']))
    Prelu_5_output = np.array(interpreter.get_tensor(details[78]['index']))
    Conv2D_block_4_Conv2D_0_weight = np.array(interpreter.get_tensor(details[79]['index']))
    Conv2D_block_4_Conv2D_0_bias = np.array(interpreter.get_tensor(details[80]['index']))
    Conv2D_block_4_Conv2D_0_output = np.array(interpreter.get_tensor(details[81]['index']))
    Conv2D_block_4_Prelu_0_weight = np.array(interpreter.get_tensor(details[82]['index']))
    Conv2D_block_4_Prelu_0_output = np.array(interpreter.get_tensor(details[83]['index']))
    Conv2D_block_4_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[84]['index']))
    Conv2D_block_4_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[85]['index']))
    Conv2D_block_4_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[86]['index']))
    Conv2D_block_4_Conv2D_1_weight = np.array(interpreter.get_tensor(details[87]['index']))
    Conv2D_block_4_Conv2D_1_bias = np.array(interpreter.get_tensor(details[88]['index']))
    Conv2D_block_4_Conv2D_1_output = np.array(interpreter.get_tensor(details[89]['index']))
    Conv2D_block_4_output = np.array(interpreter.get_tensor(details[90]['index']))

    Prelu_6_weight = np.array(interpreter.get_tensor(details[91]['index']))
    Prelu_6_output = np.array(interpreter.get_tensor(details[92]['index']))
    Conv2D_block_5_Conv2D_0_weight = np.array(interpreter.get_tensor(details[93]['index']))
    Conv2D_block_5_Conv2D_0_bias = np.array(interpreter.get_tensor(details[94]['index']))
    Conv2D_block_5_Conv2D_0_output = np.array(interpreter.get_tensor(details[95]['index']))
    Conv2D_block_5_Prelu_0_weight = np.array(interpreter.get_tensor(details[96]['index']))
    Conv2D_block_5_Prelu_0_output = np.array(interpreter.get_tensor(details[97]['index']))
    Conv2D_block_5_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[98]['index']))
    Conv2D_block_5_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[99]['index']))
    Conv2D_block_5_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[100]['index']))
    Conv2D_block_5_Conv2D_1_weight = np.array(interpreter.get_tensor(details[101]['index']))
    Conv2D_block_5_Conv2D_1_bias = np.array(interpreter.get_tensor(details[102]['index']))
    Conv2D_block_5_Conv2D_1_output = np.array(interpreter.get_tensor(details[103]['index']))
    Conv2D_block_5_output = np.array(interpreter.get_tensor(details[104]['index']))

    Prelu_7_weight = np.array(interpreter.get_tensor(details[105]['index']))
    Prelu_7_output = np.array(interpreter.get_tensor(details[106]['index']))
    Conv2D_block_6_Conv2D_0_weight = np.array(interpreter.get_tensor(details[107]['index']))
    Conv2D_block_6_Conv2D_0_bias = np.array(interpreter.get_tensor(details[108]['index']))
    Conv2D_block_6_Conv2D_0_output = np.array(interpreter.get_tensor(details[109]['index']))
    Conv2D_block_6_Prelu_0_weight = np.array(interpreter.get_tensor(details[110]['index']))
    Conv2D_block_6_Prelu_0_output = np.array(interpreter.get_tensor(details[111]['index']))
    Conv2D_block_6_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[112]['index']))
    Conv2D_block_6_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[113]['index']))
    Conv2D_block_6_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[114]['index']))
    Conv2D_block_6_Conv2D_1_weight = np.array(interpreter.get_tensor(details[115]['index']))
    Conv2D_block_6_Conv2D_1_bias = np.array(interpreter.get_tensor(details[116]['index']))
    Conv2D_block_6_Conv2D_1_output = np.array(interpreter.get_tensor(details[117]['index']))
    Conv2D_block_6_output = np.array(interpreter.get_tensor(details[118]['index']))

    Prelu_8_weight = np.array(interpreter.get_tensor(details[119]['index']))
    Prelu_8_output = np.array(interpreter.get_tensor(details[120]['index']))
    Conv2D_block_7_Conv2D_0_weight = np.array(interpreter.get_tensor(details[121]['index']))
    Conv2D_block_7_Conv2D_0_bias = np.array(interpreter.get_tensor(details[122]['index']))
    Conv2D_block_7_Conv2D_0_output = np.array(interpreter.get_tensor(details[123]['index']))
    Conv2D_block_7_Prelu_0_weight = np.array(interpreter.get_tensor(details[124]['index']))
    Conv2D_block_7_Prelu_0_output = np.array(interpreter.get_tensor(details[125]['index']))
    Conv2D_block_7_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[126]['index']))
    Conv2D_block_7_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[127]['index']))
    Conv2D_block_7_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[128]['index']))
    Conv2D_block_7_Conv2D_1_weight = np.array(interpreter.get_tensor(details[129]['index']))
    Conv2D_block_7_Conv2D_1_bias = np.array(interpreter.get_tensor(details[130]['index']))
    Conv2D_block_7_Conv2D_1_output = np.array(interpreter.get_tensor(details[131]['index']))
    Conv2D_block_7_output = np.array(interpreter.get_tensor(details[132]['index']))

    Prelu_9_weight = np.array(interpreter.get_tensor(details[133]['index']))
    Prelu_9_output = np.array(interpreter.get_tensor(details[134]['index']))
    Conv2D_block_v2_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[135]['index']))
    Conv2D_block_v2_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[136]['index']))
    Conv2D_block_v2_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[137]['index']))
    Conv2D_block_v2_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[138]['index']))
    Conv2D_block_v2_1_Prelu_0_output = np.array(interpreter.get_tensor(details[139]['index']))
    Conv2D_block_v2_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[140]['index']))
    Conv2D_block_v2_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[141]['index']))
    Conv2D_block_v2_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[142]['index']))
    Conv2D_block_v2_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[143]['index']))
    Conv2D_block_v2_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[144]['index']))
    Conv2D_block_v2_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[145]['index']))
    Conv2D_block_v2_1_maxpool_output = np.array(interpreter.get_tensor(details[146]['index']))
    Conv2D_block_v2_1_output = np.array(interpreter.get_tensor(details[147]['index']))
    Prelu_10_weight = np.array(interpreter.get_tensor(details[148]['index']))
    Prelu_10_output = np.array(interpreter.get_tensor(details[149]['index']))

    eyeContour_Conv2D_block_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[150]['index']))
    eyeContour_Conv2D_block_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[151]['index']))
    eyeContour_Conv2D_block_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[152]['index']))
    eyeContour_Conv2D_block_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[156]['index']))
    eyeContour_Conv2D_block_0_Prelu_0_output = np.array(interpreter.get_tensor(details[157]['index']))
    eyeContour_Conv2D_block_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[160]['index']))
    eyeContour_Conv2D_block_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[161]['index']))
    eyeContour_Conv2D_block_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[162]['index']))
    eyeContour_Conv2D_block_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[166]['index']))
    eyeContour_Conv2D_block_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[167]['index']))
    eyeContour_Conv2D_block_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[168]['index']))
    eyeContour_Conv2D_block_0_output = np.array(interpreter.get_tensor(details[172]['index']))

    eyeContour_Prelu_1_weight = np.array(interpreter.get_tensor(details[174]['index']))
    eyeContour_Prelu_1_output = np.array(interpreter.get_tensor(details[175]['index']))
    eyeContour_Conv2D_block_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[178]['index']))
    eyeContour_Conv2D_block_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[179]['index']))
    eyeContour_Conv2D_block_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[180]['index']))
    eyeContour_Conv2D_block_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[184]['index']))
    eyeContour_Conv2D_block_1_Prelu_0_output = np.array(interpreter.get_tensor(details[185]['index']))
    eyeContour_Conv2D_block_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[188]['index']))
    eyeContour_Conv2D_block_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[189]['index']))
    eyeContour_Conv2D_block_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[190]['index']))
    eyeContour_Conv2D_block_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[194]['index']))
    eyeContour_Conv2D_block_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[195]['index']))
    eyeContour_Conv2D_block_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[196]['index']))
    eyeContour_Conv2D_block_1_output = np.array(interpreter.get_tensor(details[200]['index']))

    eyeContour_Prelu_2_weight = np.array(interpreter.get_tensor(details[202]['index']))
    eyeContour_Prelu_2_output = np.array(interpreter.get_tensor(details[203]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[206]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[207]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[208]['index']))
    eyeContour_Conv2D_block_v2_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[212]['index']))
    eyeContour_Conv2D_block_v2_0_Prelu_0_output = np.array(interpreter.get_tensor(details[213]['index']))
    eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[216]['index']))
    eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[217]['index']))
    eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[218]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[222]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[223]['index']))
    eyeContour_Conv2D_block_v2_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[224]['index']))
    eyeContour_Conv2D_block_v2_0_maxpool_output = np.array(interpreter.get_tensor(details[228]['index']))
    eyeContour_Conv2D_block_v2_0_output = np.array(interpreter.get_tensor(details[230]['index']))

    eyeContour_Prelu_3_weight = np.array(interpreter.get_tensor(details[232]['index']))
    eyeContour_Prelu_3_output = np.array(interpreter.get_tensor(details[233]['index']))
    eyeContour_Conv2D_block_2_Conv2D_0_weight = np.array(interpreter.get_tensor(details[236]['index']))
    eyeContour_Conv2D_block_2_Conv2D_0_bias = np.array(interpreter.get_tensor(details[237]['index']))
    eyeContour_Conv2D_block_2_Conv2D_0_output = np.array(interpreter.get_tensor(details[238]['index']))
    eyeContour_Conv2D_block_2_Prelu_0_weight = np.array(interpreter.get_tensor(details[242]['index']))
    eyeContour_Conv2D_block_2_Prelu_0_output = np.array(interpreter.get_tensor(details[243]['index']))
    eyeContour_Conv2D_block_2_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[246]['index']))
    eyeContour_Conv2D_block_2_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[247]['index']))
    eyeContour_Conv2D_block_2_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[248]['index']))
    eyeContour_Conv2D_block_2_Conv2D_1_weight = np.array(interpreter.get_tensor(details[252]['index']))
    eyeContour_Conv2D_block_2_Conv2D_1_bias = np.array(interpreter.get_tensor(details[253]['index']))
    eyeContour_Conv2D_block_2_Conv2D_1_output = np.array(interpreter.get_tensor(details[254]['index']))
    eyeContour_Conv2D_block_2_output = np.array(interpreter.get_tensor(details[258]['index']))

    eyeContour_Prelu_4_weight = np.array(interpreter.get_tensor(details[260]['index']))
    eyeContour_Prelu_4_output = np.array(interpreter.get_tensor(details[261]['index']))
    eyeContour_Conv2D_block_3_Conv2D_0_weight = np.array(interpreter.get_tensor(details[264]['index']))
    eyeContour_Conv2D_block_3_Conv2D_0_bias = np.array(interpreter.get_tensor(details[265]['index']))
    eyeContour_Conv2D_block_3_Conv2D_0_output = np.array(interpreter.get_tensor(details[266]['index']))
    eyeContour_Conv2D_block_3_Prelu_0_weight = np.array(interpreter.get_tensor(details[270]['index']))
    eyeContour_Conv2D_block_3_Prelu_0_output = np.array(interpreter.get_tensor(details[271]['index']))
    eyeContour_Conv2D_block_3_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[274]['index']))
    eyeContour_Conv2D_block_3_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[275]['index']))
    eyeContour_Conv2D_block_3_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[276]['index']))
    eyeContour_Conv2D_block_3_Conv2D_1_weight = np.array(interpreter.get_tensor(details[280]['index']))
    eyeContour_Conv2D_block_3_Conv2D_1_bias = np.array(interpreter.get_tensor(details[281]['index']))
    eyeContour_Conv2D_block_3_Conv2D_1_output = np.array(interpreter.get_tensor(details[282]['index']))
    eyeContour_Conv2D_block_3_output = np.array(interpreter.get_tensor(details[286]['index']))

    eyeContour_Prelu_5_weight = np.array(interpreter.get_tensor(details[288]['index']))
    eyeContour_Prelu_5_output = np.array(interpreter.get_tensor(details[289]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[292]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[293]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[294]['index']))
    eyeContour_Conv2D_block_v2_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[298]['index']))
    eyeContour_Conv2D_block_v2_1_Prelu_0_output = np.array(interpreter.get_tensor(details[299]['index']))
    eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[302]['index']))
    eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[303]['index']))
    eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[304]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[308]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[309]['index']))
    eyeContour_Conv2D_block_v2_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[310]['index']))
    eyeContour_Conv2D_block_v2_1_maxpool_output = np.array(interpreter.get_tensor(details[314]['index']))
    eyeContour_Conv2D_block_v2_1_output = np.array(interpreter.get_tensor(details[316]['index']))

    eyeContour_Prelu_6_weight = np.array(interpreter.get_tensor(details[318]['index']))
    eyeContour_Prelu_6_output = np.array(interpreter.get_tensor(details[319]['index']))
    eyeContour_Conv2D_block_4_Conv2D_0_weight = np.array(interpreter.get_tensor(details[322]['index']))
    eyeContour_Conv2D_block_4_Conv2D_0_bias = np.array(interpreter.get_tensor(details[323]['index']))
    eyeContour_Conv2D_block_4_Conv2D_0_output = np.array(interpreter.get_tensor(details[324]['index']))
    eyeContour_Conv2D_block_4_Prelu_0_weight = np.array(interpreter.get_tensor(details[328]['index']))
    eyeContour_Conv2D_block_4_Prelu_0_output = np.array(interpreter.get_tensor(details[329]['index']))
    eyeContour_Conv2D_block_4_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[332]['index']))
    eyeContour_Conv2D_block_4_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[333]['index']))
    eyeContour_Conv2D_block_4_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[334]['index']))
    eyeContour_Conv2D_block_4_Conv2D_1_weight = np.array(interpreter.get_tensor(details[338]['index']))
    eyeContour_Conv2D_block_4_Conv2D_1_bias = np.array(interpreter.get_tensor(details[339]['index']))
    eyeContour_Conv2D_block_4_Conv2D_1_output = np.array(interpreter.get_tensor(details[340]['index']))
    eyeContour_Conv2D_block_4_output = np.array(interpreter.get_tensor(details[344]['index']))

    eyeContour_Prelu_7_weight = np.array(interpreter.get_tensor(details[346]['index']))
    eyeContour_Prelu_7_output = np.array(interpreter.get_tensor(details[347]['index']))
    eyeContour_Conv2D_block_5_Conv2D_0_weight = np.array(interpreter.get_tensor(details[350]['index']))
    eyeContour_Conv2D_block_5_Conv2D_0_bias = np.array(interpreter.get_tensor(details[351]['index']))
    eyeContour_Conv2D_block_5_Conv2D_0_output = np.array(interpreter.get_tensor(details[352]['index']))
    eyeContour_Conv2D_block_5_Prelu_0_weight = np.array(interpreter.get_tensor(details[356]['index']))
    eyeContour_Conv2D_block_5_Prelu_0_output = np.array(interpreter.get_tensor(details[357]['index']))
    eyeContour_Conv2D_block_5_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[360]['index']))
    eyeContour_Conv2D_block_5_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[361]['index']))
    eyeContour_Conv2D_block_5_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[362]['index']))
    eyeContour_Conv2D_block_5_Conv2D_1_weight = np.array(interpreter.get_tensor(details[366]['index']))
    eyeContour_Conv2D_block_5_Conv2D_1_bias = np.array(interpreter.get_tensor(details[367]['index']))
    eyeContour_Conv2D_block_5_Conv2D_1_output = np.array(interpreter.get_tensor(details[368]['index']))
    eyeContour_Conv2D_block_5_output = np.array(interpreter.get_tensor(details[372]['index']))

    eyeContour_Prelu_8_weight = np.array(interpreter.get_tensor(details[374]['index']))
    eyeContour_Prelu_8_output = np.array(interpreter.get_tensor(details[375]['index']))
    eyeContour_Conv2D_out_weight = np.array(interpreter.get_tensor(details[378]['index']))
    eyeContour_Conv2D_out_bias = np.array(interpreter.get_tensor(details[379]['index']))
    eyeContour_Conv2D_out_output = np.array(interpreter.get_tensor(details[380]['index']))


    iris_Conv2D_block_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[153]['index']))
    iris_Conv2D_block_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[154]['index']))
    iris_Conv2D_block_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[155]['index']))
    iris_Conv2D_block_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[158]['index']))
    iris_Conv2D_block_0_Prelu_0_output = np.array(interpreter.get_tensor(details[159]['index']))
    iris_Conv2D_block_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[163]['index']))
    iris_Conv2D_block_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[164]['index']))
    iris_Conv2D_block_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[165]['index']))
    iris_Conv2D_block_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[169]['index']))
    iris_Conv2D_block_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[170]['index']))
    iris_Conv2D_block_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[171]['index']))
    iris_Conv2D_block_0_output = np.array(interpreter.get_tensor(details[173]['index']))

    iris_Prelu_1_weight = np.array(interpreter.get_tensor(details[176]['index']))
    iris_Prelu_1_output = np.array(interpreter.get_tensor(details[177]['index']))
    iris_Conv2D_block_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[181]['index']))
    iris_Conv2D_block_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[182]['index']))
    iris_Conv2D_block_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[183]['index']))
    iris_Conv2D_block_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[186]['index']))
    iris_Conv2D_block_1_Prelu_0_output = np.array(interpreter.get_tensor(details[187]['index']))
    iris_Conv2D_block_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[191]['index']))
    iris_Conv2D_block_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[192]['index']))
    iris_Conv2D_block_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[193]['index']))
    iris_Conv2D_block_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[197]['index']))
    iris_Conv2D_block_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[198]['index']))
    iris_Conv2D_block_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[199]['index']))
    iris_Conv2D_block_1_output = np.array(interpreter.get_tensor(details[201]['index']))

    iris_Prelu_2_weight = np.array(interpreter.get_tensor(details[204]['index']))
    iris_Prelu_2_output = np.array(interpreter.get_tensor(details[205]['index']))    
    iris_Conv2D_block_v2_0_Conv2D_0_weight = np.array(interpreter.get_tensor(details[209]['index']))
    iris_Conv2D_block_v2_0_Conv2D_0_bias = np.array(interpreter.get_tensor(details[210]['index']))
    iris_Conv2D_block_v2_0_Conv2D_0_output = np.array(interpreter.get_tensor(details[211]['index']))
    iris_Conv2D_block_v2_0_Prelu_0_weight = np.array(interpreter.get_tensor(details[214]['index']))
    iris_Conv2D_block_v2_0_Prelu_0_output = np.array(interpreter.get_tensor(details[215]['index']))
    iris_Conv2D_block_v2_0_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[219]['index']))
    iris_Conv2D_block_v2_0_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[220]['index']))
    iris_Conv2D_block_v2_0_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[221]['index']))
    iris_Conv2D_block_v2_0_Conv2D_1_weight = np.array(interpreter.get_tensor(details[225]['index']))
    iris_Conv2D_block_v2_0_Conv2D_1_bias = np.array(interpreter.get_tensor(details[226]['index']))
    iris_Conv2D_block_v2_0_Conv2D_1_output = np.array(interpreter.get_tensor(details[227]['index']))
    iris_Conv2D_block_v2_0_maxpool_output = np.array(interpreter.get_tensor(details[229]['index']))
    iris_Conv2D_block_v2_0_output = np.array(interpreter.get_tensor(details[231]['index']))

    iris_Prelu_3_weight = np.array(interpreter.get_tensor(details[234]['index']))
    iris_Prelu_3_output = np.array(interpreter.get_tensor(details[235]['index']))
    iris_Conv2D_block_2_Conv2D_0_weight = np.array(interpreter.get_tensor(details[239]['index']))
    iris_Conv2D_block_2_Conv2D_0_bias = np.array(interpreter.get_tensor(details[240]['index']))
    iris_Conv2D_block_2_Conv2D_0_output = np.array(interpreter.get_tensor(details[241]['index']))
    iris_Conv2D_block_2_Prelu_0_weight = np.array(interpreter.get_tensor(details[244]['index']))
    iris_Conv2D_block_2_Prelu_0_output = np.array(interpreter.get_tensor(details[245]['index']))
    iris_Conv2D_block_2_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[249]['index']))
    iris_Conv2D_block_2_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[250]['index']))
    iris_Conv2D_block_2_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[251]['index']))
    iris_Conv2D_block_2_Conv2D_1_weight = np.array(interpreter.get_tensor(details[255]['index']))
    iris_Conv2D_block_2_Conv2D_1_bias = np.array(interpreter.get_tensor(details[256]['index']))
    iris_Conv2D_block_2_Conv2D_1_output = np.array(interpreter.get_tensor(details[257]['index']))
    iris_Conv2D_block_2_output = np.array(interpreter.get_tensor(details[259]['index']))

    iris_Prelu_4_weight = np.array(interpreter.get_tensor(details[262]['index']))
    iris_Prelu_4_output = np.array(interpreter.get_tensor(details[263]['index']))
    iris_Conv2D_block_3_Conv2D_0_weight = np.array(interpreter.get_tensor(details[267]['index']))
    iris_Conv2D_block_3_Conv2D_0_bias = np.array(interpreter.get_tensor(details[268]['index']))
    iris_Conv2D_block_3_Conv2D_0_output = np.array(interpreter.get_tensor(details[269]['index']))
    iris_Conv2D_block_3_Prelu_0_weight = np.array(interpreter.get_tensor(details[272]['index']))
    iris_Conv2D_block_3_Prelu_0_output = np.array(interpreter.get_tensor(details[273]['index']))
    iris_Conv2D_block_3_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[277]['index']))
    iris_Conv2D_block_3_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[278]['index']))
    iris_Conv2D_block_3_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[279]['index']))
    iris_Conv2D_block_3_Conv2D_1_weight = np.array(interpreter.get_tensor(details[283]['index']))
    iris_Conv2D_block_3_Conv2D_1_bias = np.array(interpreter.get_tensor(details[284]['index']))
    iris_Conv2D_block_3_Conv2D_1_output = np.array(interpreter.get_tensor(details[285]['index']))
    iris_Conv2D_block_3_output = np.array(interpreter.get_tensor(details[287]['index']))

    iris_Prelu_5_weight = np.array(interpreter.get_tensor(details[290]['index']))
    iris_Prelu_5_output = np.array(interpreter.get_tensor(details[291]['index']))
    iris_Conv2D_block_v2_1_Conv2D_0_weight = np.array(interpreter.get_tensor(details[295]['index']))
    iris_Conv2D_block_v2_1_Conv2D_0_bias = np.array(interpreter.get_tensor(details[296]['index']))
    iris_Conv2D_block_v2_1_Conv2D_0_output = np.array(interpreter.get_tensor(details[297]['index']))
    iris_Conv2D_block_v2_1_Prelu_0_weight = np.array(interpreter.get_tensor(details[300]['index']))
    iris_Conv2D_block_v2_1_Prelu_0_output = np.array(interpreter.get_tensor(details[301]['index']))
    iris_Conv2D_block_v2_1_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[305]['index']))
    iris_Conv2D_block_v2_1_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[306]['index']))
    iris_Conv2D_block_v2_1_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[307]['index']))
    iris_Conv2D_block_v2_1_Conv2D_1_weight = np.array(interpreter.get_tensor(details[311]['index']))
    iris_Conv2D_block_v2_1_Conv2D_1_bias = np.array(interpreter.get_tensor(details[312]['index']))
    iris_Conv2D_block_v2_1_Conv2D_1_output = np.array(interpreter.get_tensor(details[313]['index']))
    iris_Conv2D_block_v2_1_maxpool_output = np.array(interpreter.get_tensor(details[315]['index']))
    iris_Conv2D_block_v2_1_output = np.array(interpreter.get_tensor(details[317]['index']))

    iris_Prelu_6_weight = np.array(interpreter.get_tensor(details[320]['index']))
    iris_Prelu_6_output = np.array(interpreter.get_tensor(details[321]['index']))
    iris_Conv2D_block_4_Conv2D_0_weight = np.array(interpreter.get_tensor(details[325]['index']))
    iris_Conv2D_block_4_Conv2D_0_bias = np.array(interpreter.get_tensor(details[326]['index']))
    iris_Conv2D_block_4_Conv2D_0_output = np.array(interpreter.get_tensor(details[327]['index']))
    iris_Conv2D_block_4_Prelu_0_weight = np.array(interpreter.get_tensor(details[330]['index']))
    iris_Conv2D_block_4_Prelu_0_output = np.array(interpreter.get_tensor(details[331]['index']))
    iris_Conv2D_block_4_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[335]['index']))
    iris_Conv2D_block_4_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[336]['index']))
    iris_Conv2D_block_4_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[337]['index']))
    iris_Conv2D_block_4_Conv2D_1_weight = np.array(interpreter.get_tensor(details[341]['index']))
    iris_Conv2D_block_4_Conv2D_1_bias = np.array(interpreter.get_tensor(details[342]['index']))
    iris_Conv2D_block_4_Conv2D_1_output = np.array(interpreter.get_tensor(details[343]['index']))
    iris_Conv2D_block_4_output = np.array(interpreter.get_tensor(details[345]['index']))

    iris_Prelu_7_weight = np.array(interpreter.get_tensor(details[348]['index']))
    iris_Prelu_7_output = np.array(interpreter.get_tensor(details[349]['index']))
    iris_Conv2D_block_5_Conv2D_0_weight = np.array(interpreter.get_tensor(details[353]['index']))
    iris_Conv2D_block_5_Conv2D_0_bias = np.array(interpreter.get_tensor(details[354]['index']))
    iris_Conv2D_block_5_Conv2D_0_output = np.array(interpreter.get_tensor(details[355]['index']))
    iris_Conv2D_block_5_Prelu_0_weight = np.array(interpreter.get_tensor(details[358]['index']))
    iris_Conv2D_block_5_Prelu_0_output = np.array(interpreter.get_tensor(details[359]['index']))
    iris_Conv2D_block_5_DepthwiseConv2d_weight = np.array(interpreter.get_tensor(details[363]['index']))
    iris_Conv2D_block_5_DepthwiseConv2d_bias = np.array(interpreter.get_tensor(details[364]['index']))
    iris_Conv2D_block_5_DepthwiseConv2d_output = np.array(interpreter.get_tensor(details[365]['index']))
    iris_Conv2D_block_5_Conv2D_1_weight = np.array(interpreter.get_tensor(details[369]['index']))
    iris_Conv2D_block_5_Conv2D_1_bias = np.array(interpreter.get_tensor(details[370]['index']))
    iris_Conv2D_block_5_Conv2D_1_output = np.array(interpreter.get_tensor(details[371]['index']))
    iris_Conv2D_block_5_output = np.array(interpreter.get_tensor(details[373]['index']))

    iris_Prelu_8_weight = np.array(interpreter.get_tensor(details[376]['index']))
    iris_Prelu_8_output = np.array(interpreter.get_tensor(details[377]['index']))
    iris_Conv2D_out_weight = np.array(interpreter.get_tensor(details[381]['index']))
    iris_Conv2D_out_bias = np.array(interpreter.get_tensor(details[382]['index']))
    iris_Conv2D_out_output = np.array(interpreter.get_tensor(details[383]['index']))

    weights = {
        "input": tfLite_weight_to_torch(input),
        "Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_0_weight),
        "Conv2D_0_bias": Conv2D_0_bias,
        "Conv2D_0_output": tfLite_weight_to_torch(Conv2D_0_output),
        
        "Prelu_0_weight": Prelu_0_weight.flatten(),
        "Prelu_0_output": tfLite_weight_to_torch(Prelu_0_output),
        "Conv2D_block_0_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_0_Conv2D_0_weight),
        "Conv2D_block_0_Conv2D_0_bias": Conv2D_block_0_Conv2D_0_bias.flatten(),
        "Conv2D_block_0_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_0_Conv2D_0_output),
        "Conv2D_block_0_Prelu_0_weight": Conv2D_block_0_Prelu_0_weight.flatten(),
        "Conv2D_block_0_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_0_Prelu_0_output),
        "Conv2D_block_0_DepthwiseConv2d_weight": np.transpose(Conv2D_block_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_0_DepthwiseConv2d_bias": Conv2D_block_0_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_0_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_0_DepthwiseConv2d_output),
        "Conv2D_block_0_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_0_Conv2D_1_weight),
        "Conv2D_block_0_Conv2D_1_bias": Conv2D_block_0_Conv2D_1_bias.flatten(),
        "Conv2D_block_0_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_0_Conv2D_1_output),
        "Conv2D_block_0_output": tfLite_weight_to_torch(Conv2D_block_0_output),
        
        "Prelu_1_weight": Prelu_1_weight.flatten(),
        "Prelu_1_output": tfLite_weight_to_torch(Prelu_1_output),
        "Conv2D_block_1_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_1_Conv2D_0_weight),
        "Conv2D_block_1_Conv2D_0_bias": Conv2D_block_1_Conv2D_0_bias.flatten(),
        "Conv2D_block_1_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_1_Conv2D_0_output),
        "Conv2D_block_1_Prelu_0_weight": Conv2D_block_1_Prelu_0_weight.flatten(),
        "Conv2D_block_1_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_1_Prelu_0_output),
        "Conv2D_block_1_DepthwiseConv2d_weight": np.transpose(Conv2D_block_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_1_DepthwiseConv2d_bias": Conv2D_block_1_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_1_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_1_DepthwiseConv2d_output),
        "Conv2D_block_1_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_1_Conv2D_1_weight),
        "Conv2D_block_1_Conv2D_1_bias": Conv2D_block_1_Conv2D_1_bias.flatten(),
        "Conv2D_block_1_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_1_Conv2D_1_output),
        "Conv2D_block_1_output": tfLite_weight_to_torch(Conv2D_block_1_output),

        "Prelu_2_weight": Prelu_2_weight.flatten(),
        "Prelu_2_output": tfLite_weight_to_torch(Prelu_2_output),
        "Conv2D_block_2_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_2_Conv2D_0_weight),
        "Conv2D_block_2_Conv2D_0_bias": Conv2D_block_2_Conv2D_0_bias.flatten(),
        "Conv2D_block_2_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_2_Conv2D_0_output),
        "Conv2D_block_2_Prelu_0_weight": Conv2D_block_2_Prelu_0_weight.flatten(),
        "Conv2D_block_2_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_2_Prelu_0_output),
        "Conv2D_block_2_DepthwiseConv2d_weight": np.transpose(Conv2D_block_2_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_2_DepthwiseConv2d_bias": Conv2D_block_2_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_2_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_2_DepthwiseConv2d_output),
        "Conv2D_block_2_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_2_Conv2D_1_weight),
        "Conv2D_block_2_Conv2D_1_bias": Conv2D_block_2_Conv2D_1_bias.flatten(),
        "Conv2D_block_2_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_2_Conv2D_1_output),
        "Conv2D_block_2_output": tfLite_weight_to_torch(Conv2D_block_2_output),

        "Prelu_3_weight": Prelu_3_weight.flatten(),
        "Prelu_3_output": tfLite_weight_to_torch(Prelu_3_output),
        "Conv2D_block_3_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_3_Conv2D_0_weight),
        "Conv2D_block_3_Conv2D_0_bias": Conv2D_block_3_Conv2D_0_bias.flatten(),
        "Conv2D_block_3_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_3_Conv2D_0_output),
        "Conv2D_block_3_Prelu_0_weight": Conv2D_block_3_Prelu_0_weight.flatten(),
        "Conv2D_block_3_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_3_Prelu_0_output),
        "Conv2D_block_3_DepthwiseConv2d_weight": np.transpose(Conv2D_block_3_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_3_DepthwiseConv2d_bias": Conv2D_block_3_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_3_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_3_DepthwiseConv2d_output),
        "Conv2D_block_3_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_3_Conv2D_1_weight),
        "Conv2D_block_3_Conv2D_1_bias": Conv2D_block_3_Conv2D_1_bias.flatten(),
        "Conv2D_block_3_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_3_Conv2D_1_output),
        "Conv2D_block_3_output": tfLite_weight_to_torch(Conv2D_block_3_output),

        "Prelu_4_weight": Prelu_4_weight.flatten(),
        "Prelu_4_output": tfLite_weight_to_torch(Prelu_4_output),
        "Conv2D_block_v2_0_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_v2_0_Conv2D_0_weight),
        "Conv2D_block_v2_0_Conv2D_0_bias": Conv2D_block_v2_0_Conv2D_0_bias.flatten(),
        "Conv2D_block_v2_0_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_v2_0_Conv2D_0_output),
        "Conv2D_block_v2_0_Prelu_0_weight": Conv2D_block_v2_0_Prelu_0_weight.flatten(),
        "Conv2D_block_v2_0_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_v2_0_Prelu_0_output),
        "Conv2D_block_v2_0_DepthwiseConv2d_weight": np.transpose(Conv2D_block_v2_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_v2_0_DepthwiseConv2d_bias": Conv2D_block_v2_0_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_v2_0_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_v2_0_DepthwiseConv2d_output),
        "Conv2D_block_v2_0_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_v2_0_Conv2D_1_weight),
        "Conv2D_block_v2_0_Conv2D_1_bias": Conv2D_block_v2_0_Conv2D_1_bias.flatten(),
        "Conv2D_block_v2_0_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_v2_0_Conv2D_1_output),
        "Conv2D_block_v2_0_maxpool_output": tfLite_weight_to_torch(Conv2D_block_v2_0_maxpool_output),
        "Conv2D_block_v2_0_padding": Conv2D_block_v2_0_padding,
        "Conv2D_block_v2_0_padding_output": tfLite_weight_to_torch(Conv2D_block_v2_0_padding_output),
        "Conv2D_block_v2_0_output": tfLite_weight_to_torch(Conv2D_block_v2_0_output),

        "Prelu_5_weight": Prelu_5_weight.flatten(),
        "Prelu_5_output": tfLite_weight_to_torch(Prelu_5_output),
        "Conv2D_block_4_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_4_Conv2D_0_weight),
        "Conv2D_block_4_Conv2D_0_bias": Conv2D_block_4_Conv2D_0_bias.flatten(),
        "Conv2D_block_4_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_4_Conv2D_0_output),
        "Conv2D_block_4_Prelu_0_weight": Conv2D_block_4_Prelu_0_weight.flatten(),
        "Conv2D_block_4_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_4_Prelu_0_output),
        "Conv2D_block_4_DepthwiseConv2d_weight": np.transpose(Conv2D_block_4_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_4_DepthwiseConv2d_bias": Conv2D_block_4_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_4_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_4_DepthwiseConv2d_output),
        "Conv2D_block_4_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_4_Conv2D_1_weight),
        "Conv2D_block_4_Conv2D_1_bias": Conv2D_block_4_Conv2D_1_bias.flatten(),
        "Conv2D_block_4_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_4_Conv2D_1_output),
        "Conv2D_block_4_output": tfLite_weight_to_torch(Conv2D_block_4_output),

        "Prelu_6_weight": Prelu_6_weight.flatten(),
        "Prelu_6_output": tfLite_weight_to_torch(Prelu_6_output),
        "Conv2D_block_5_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_5_Conv2D_0_weight),
        "Conv2D_block_5_Conv2D_0_bias": Conv2D_block_5_Conv2D_0_bias.flatten(),
        "Conv2D_block_5_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_5_Conv2D_0_output),
        "Conv2D_block_5_Prelu_0_weight": Conv2D_block_5_Prelu_0_weight.flatten(),
        "Conv2D_block_5_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_5_Prelu_0_output),
        "Conv2D_block_5_DepthwiseConv2d_weight": np.transpose(Conv2D_block_5_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_5_DepthwiseConv2d_bias": Conv2D_block_5_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_5_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_5_DepthwiseConv2d_output),
        "Conv2D_block_5_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_5_Conv2D_1_weight),
        "Conv2D_block_5_Conv2D_1_bias": Conv2D_block_5_Conv2D_1_bias.flatten(),
        "Conv2D_block_5_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_5_Conv2D_1_output),
        "Conv2D_block_5_output": tfLite_weight_to_torch(Conv2D_block_5_output),
    
        "Prelu_7_weight": Prelu_7_weight.flatten(),
        "Prelu_7_output": tfLite_weight_to_torch(Prelu_7_output),
        "Conv2D_block_6_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_6_Conv2D_0_weight),
        "Conv2D_block_6_Conv2D_0_bias": Conv2D_block_6_Conv2D_0_bias.flatten(),
        "Conv2D_block_6_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_6_Conv2D_0_output),
        "Conv2D_block_6_Prelu_0_weight": Conv2D_block_6_Prelu_0_weight.flatten(),
        "Conv2D_block_6_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_6_Prelu_0_output),
        "Conv2D_block_6_DepthwiseConv2d_weight": np.transpose(Conv2D_block_6_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_6_DepthwiseConv2d_bias": Conv2D_block_6_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_6_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_6_DepthwiseConv2d_output),
        "Conv2D_block_6_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_6_Conv2D_1_weight),
        "Conv2D_block_6_Conv2D_1_bias": Conv2D_block_6_Conv2D_1_bias.flatten(),
        "Conv2D_block_6_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_6_Conv2D_1_output),
        "Conv2D_block_6_output": tfLite_weight_to_torch(Conv2D_block_6_output),

        "Prelu_8_weight": Prelu_8_weight.flatten(),
        "Prelu_8_output": tfLite_weight_to_torch(Prelu_8_output),
        "Conv2D_block_7_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_7_Conv2D_0_weight),
        "Conv2D_block_7_Conv2D_0_bias": Conv2D_block_7_Conv2D_0_bias.flatten(),
        "Conv2D_block_7_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_7_Conv2D_0_output),
        "Conv2D_block_7_Prelu_0_weight": Conv2D_block_7_Prelu_0_weight.flatten(),
        "Conv2D_block_7_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_7_Prelu_0_output),
        "Conv2D_block_7_DepthwiseConv2d_weight": np.transpose(Conv2D_block_7_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_7_DepthwiseConv2d_bias": Conv2D_block_7_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_7_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_7_DepthwiseConv2d_output),
        "Conv2D_block_7_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_7_Conv2D_1_weight),
        "Conv2D_block_7_Conv2D_1_bias": Conv2D_block_7_Conv2D_1_bias.flatten(),
        "Conv2D_block_7_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_7_Conv2D_1_output),
        "Conv2D_block_7_output": tfLite_weight_to_torch(Conv2D_block_7_output),

        "Prelu_9_weight": Prelu_9_weight.flatten(),
        "Prelu_9_output": tfLite_weight_to_torch(Prelu_9_output),
        "Conv2D_block_v2_1_Conv2D_0_weight": tfLite_weight_to_torch(Conv2D_block_v2_1_Conv2D_0_weight),
        "Conv2D_block_v2_1_Conv2D_0_bias": Conv2D_block_v2_1_Conv2D_0_bias.flatten(),
        "Conv2D_block_v2_1_Conv2D_0_output": tfLite_weight_to_torch(Conv2D_block_v2_1_Conv2D_0_output),
        "Conv2D_block_v2_1_Prelu_0_weight": Conv2D_block_v2_1_Prelu_0_weight.flatten(),
        "Conv2D_block_v2_1_Prelu_0_output": tfLite_weight_to_torch(Conv2D_block_v2_1_Prelu_0_output),
        "Conv2D_block_v2_1_DepthwiseConv2d_weight": np.transpose(Conv2D_block_v2_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "Conv2D_block_v2_1_DepthwiseConv2d_bias": Conv2D_block_v2_1_DepthwiseConv2d_bias.flatten(),
        "Conv2D_block_v2_1_DepthwiseConv2d_output": tfLite_weight_to_torch(Conv2D_block_v2_1_DepthwiseConv2d_output),
        "Conv2D_block_v2_1_Conv2D_1_weight": tfLite_weight_to_torch(Conv2D_block_v2_1_Conv2D_1_weight),
        "Conv2D_block_v2_1_Conv2D_1_bias": Conv2D_block_v2_1_Conv2D_1_bias.flatten(),
        "Conv2D_block_v2_1_Conv2D_1_output": tfLite_weight_to_torch(Conv2D_block_v2_1_Conv2D_1_output),
        "Conv2D_block_v2_1_maxpool_output": tfLite_weight_to_torch(Conv2D_block_v2_1_maxpool_output),
        "Conv2D_block_v2_1_output": tfLite_weight_to_torch(Conv2D_block_v2_1_output),

        "Prelu_10_weight": Prelu_10_weight.flatten(),
        "Prelu_10_output": tfLite_weight_to_torch(Prelu_10_output),

        "eyeContour_Conv2D_block_0_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_Conv2D_0_weight),
        "eyeContour_Conv2D_block_0_Conv2D_0_bias": eyeContour_Conv2D_block_0_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_0_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_Conv2D_0_output),
        "eyeContour_Conv2D_block_0_Prelu_0_weight": eyeContour_Conv2D_block_0_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_0_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_Prelu_0_output),
        "eyeContour_Conv2D_block_0_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_0_DepthwiseConv2d_bias": eyeContour_Conv2D_block_0_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_0_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_0_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_Conv2D_1_weight),
        "eyeContour_Conv2D_block_0_Conv2D_1_bias": eyeContour_Conv2D_block_0_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_0_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_Conv2D_1_output),
        "eyeContour_Conv2D_block_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_0_output),

        "eyeContour_Prelu_1_weight": eyeContour_Prelu_1_weight.flatten(),
        "eyeContour_Prelu_1_output": tfLite_weight_to_torch(eyeContour_Prelu_1_output),
        "eyeContour_Conv2D_block_1_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_Conv2D_0_weight),
        "eyeContour_Conv2D_block_1_Conv2D_0_bias": eyeContour_Conv2D_block_1_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_1_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_Conv2D_0_output),
        "eyeContour_Conv2D_block_1_Prelu_0_weight": eyeContour_Conv2D_block_1_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_1_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_Prelu_0_output),
        "eyeContour_Conv2D_block_1_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_1_DepthwiseConv2d_bias": eyeContour_Conv2D_block_1_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_1_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_1_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_Conv2D_1_weight),
        "eyeContour_Conv2D_block_1_Conv2D_1_bias": eyeContour_Conv2D_block_1_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_1_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_Conv2D_1_output),
        "eyeContour_Conv2D_block_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_1_output),

        "eyeContour_Prelu_2_weight": eyeContour_Prelu_2_weight.flatten(),
        "eyeContour_Prelu_2_output": tfLite_weight_to_torch(eyeContour_Prelu_2_output),
        "eyeContour_Conv2D_block_v2_0_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_Conv2D_0_weight),
        "eyeContour_Conv2D_block_v2_0_Conv2D_0_bias": eyeContour_Conv2D_block_v2_0_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_v2_0_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_Conv2D_0_output),
        "eyeContour_Conv2D_block_v2_0_Prelu_0_weight": eyeContour_Conv2D_block_v2_0_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_v2_0_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_Prelu_0_output),
        "eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_bias": eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_v2_0_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_Conv2D_1_weight),
        "eyeContour_Conv2D_block_v2_0_Conv2D_1_bias": eyeContour_Conv2D_block_v2_0_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_v2_0_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_Conv2D_1_output),
        "eyeContour_Conv2D_block_v2_0_maxpool_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_maxpool_output),
        "eyeContour_Conv2D_block_v2_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_0_output),

        "eyeContour_Prelu_3_weight": eyeContour_Prelu_3_weight.flatten(),
        "eyeContour_Prelu_3_output": tfLite_weight_to_torch(eyeContour_Prelu_3_output),
        "eyeContour_Conv2D_block_2_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_Conv2D_0_weight),
        "eyeContour_Conv2D_block_2_Conv2D_0_bias": eyeContour_Conv2D_block_2_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_2_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_Conv2D_0_output),
        "eyeContour_Conv2D_block_2_Prelu_0_weight": eyeContour_Conv2D_block_2_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_2_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_Prelu_0_output),
        "eyeContour_Conv2D_block_2_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_2_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_2_DepthwiseConv2d_bias": eyeContour_Conv2D_block_2_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_2_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_2_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_Conv2D_1_weight),
        "eyeContour_Conv2D_block_2_Conv2D_1_bias": eyeContour_Conv2D_block_2_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_2_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_Conv2D_1_output),
        "eyeContour_Conv2D_block_2_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_2_output),

        "eyeContour_Prelu_4_weight": eyeContour_Prelu_4_weight.flatten(),
        "eyeContour_Prelu_4_output": tfLite_weight_to_torch(eyeContour_Prelu_4_output),
        "eyeContour_Conv2D_block_3_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_Conv2D_0_weight),
        "eyeContour_Conv2D_block_3_Conv2D_0_bias": eyeContour_Conv2D_block_3_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_3_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_Conv2D_0_output),
        "eyeContour_Conv2D_block_3_Prelu_0_weight": eyeContour_Conv2D_block_3_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_3_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_Prelu_0_output),
        "eyeContour_Conv2D_block_3_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_3_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_3_DepthwiseConv2d_bias": eyeContour_Conv2D_block_3_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_3_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_3_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_Conv2D_1_weight),
        "eyeContour_Conv2D_block_3_Conv2D_1_bias": eyeContour_Conv2D_block_3_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_3_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_Conv2D_1_output),
        "eyeContour_Conv2D_block_3_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_3_output),

        "eyeContour_Prelu_5_weight": eyeContour_Prelu_5_weight.flatten(),
        "eyeContour_Prelu_5_output": tfLite_weight_to_torch(eyeContour_Prelu_5_output),
        "eyeContour_Conv2D_block_v2_1_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_Conv2D_0_weight),
        "eyeContour_Conv2D_block_v2_1_Conv2D_0_bias": eyeContour_Conv2D_block_v2_1_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_v2_1_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_Conv2D_0_output),
        "eyeContour_Conv2D_block_v2_1_Prelu_0_weight": eyeContour_Conv2D_block_v2_1_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_v2_1_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_Prelu_0_output),
        "eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_bias": eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_v2_1_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_Conv2D_1_weight),
        "eyeContour_Conv2D_block_v2_1_Conv2D_1_bias": eyeContour_Conv2D_block_v2_1_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_v2_1_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_Conv2D_1_output),
        "eyeContour_Conv2D_block_v2_1_maxpool_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_maxpool_output),
        "eyeContour_Conv2D_block_v2_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_v2_1_output),

        "eyeContour_Prelu_6_weight": eyeContour_Prelu_6_weight.flatten(),
        "eyeContour_Prelu_6_output": tfLite_weight_to_torch(eyeContour_Prelu_6_output),
        "eyeContour_Conv2D_block_4_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_Conv2D_0_weight),
        "eyeContour_Conv2D_block_4_Conv2D_0_bias": eyeContour_Conv2D_block_4_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_4_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_Conv2D_0_output),
        "eyeContour_Conv2D_block_4_Prelu_0_weight": eyeContour_Conv2D_block_4_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_4_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_Prelu_0_output),
        "eyeContour_Conv2D_block_4_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_4_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_4_DepthwiseConv2d_bias": eyeContour_Conv2D_block_4_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_4_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_4_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_Conv2D_1_weight),
        "eyeContour_Conv2D_block_4_Conv2D_1_bias": eyeContour_Conv2D_block_4_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_4_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_Conv2D_1_output),
        "eyeContour_Conv2D_block_4_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_4_output),

        "eyeContour_Prelu_7_weight": eyeContour_Prelu_7_weight.flatten(),
        "eyeContour_Prelu_7_output": tfLite_weight_to_torch(eyeContour_Prelu_7_output),
        "eyeContour_Conv2D_block_5_Conv2D_0_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_Conv2D_0_weight),
        "eyeContour_Conv2D_block_5_Conv2D_0_bias": eyeContour_Conv2D_block_5_Conv2D_0_bias.flatten(),
        "eyeContour_Conv2D_block_5_Conv2D_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_Conv2D_0_output),
        "eyeContour_Conv2D_block_5_Prelu_0_weight": eyeContour_Conv2D_block_5_Prelu_0_weight.flatten(),
        "eyeContour_Conv2D_block_5_Prelu_0_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_Prelu_0_output),
        "eyeContour_Conv2D_block_5_DepthwiseConv2d_weight": np.transpose(eyeContour_Conv2D_block_5_DepthwiseConv2d_weight, (3,0,1,2)),
        "eyeContour_Conv2D_block_5_DepthwiseConv2d_bias": eyeContour_Conv2D_block_5_DepthwiseConv2d_bias.flatten(),
        "eyeContour_Conv2D_block_5_DepthwiseConv2d_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_DepthwiseConv2d_output),
        "eyeContour_Conv2D_block_5_Conv2D_1_weight": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_Conv2D_1_weight),
        "eyeContour_Conv2D_block_5_Conv2D_1_bias": eyeContour_Conv2D_block_5_Conv2D_1_bias.flatten(),
        "eyeContour_Conv2D_block_5_Conv2D_1_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_Conv2D_1_output),
        "eyeContour_Conv2D_block_5_output": tfLite_weight_to_torch(eyeContour_Conv2D_block_5_output),

        "eyeContour_Prelu_8_weight": eyeContour_Prelu_8_weight.flatten(),
        "eyeContour_Prelu_8_output": tfLite_weight_to_torch(eyeContour_Prelu_8_output),
        "eyeContour_Conv2D_out_weight": tfLite_weight_to_torch(eyeContour_Conv2D_out_weight),
        "eyeContour_Conv2D_out_bias": eyeContour_Conv2D_out_bias.flatten(),
        "eyeContour_Conv2D_out_output": tfLite_weight_to_torch(eyeContour_Conv2D_out_output),
        "eyeContour_output": output_data_eyeContour.flatten(),

        "iris_Conv2D_block_0_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_0_Conv2D_0_weight),
        "iris_Conv2D_block_0_Conv2D_0_bias": iris_Conv2D_block_0_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_0_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_0_Conv2D_0_output),
        "iris_Conv2D_block_0_Prelu_0_weight": iris_Conv2D_block_0_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_0_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_0_Prelu_0_output),
        "iris_Conv2D_block_0_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_0_DepthwiseConv2d_bias": iris_Conv2D_block_0_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_0_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_0_DepthwiseConv2d_output),
        "iris_Conv2D_block_0_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_0_Conv2D_1_weight),
        "iris_Conv2D_block_0_Conv2D_1_bias": iris_Conv2D_block_0_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_0_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_0_Conv2D_1_output),
        "iris_Conv2D_block_0_output": tfLite_weight_to_torch(iris_Conv2D_block_0_output),

        "iris_Prelu_1_weight": iris_Prelu_1_weight.flatten(),
        "iris_Prelu_1_output": tfLite_weight_to_torch(iris_Prelu_1_output),
        "iris_Conv2D_block_1_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_1_Conv2D_0_weight),
        "iris_Conv2D_block_1_Conv2D_0_bias": iris_Conv2D_block_1_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_1_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_1_Conv2D_0_output),
        "iris_Conv2D_block_1_Prelu_0_weight": iris_Conv2D_block_1_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_1_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_1_Prelu_0_output),
        "iris_Conv2D_block_1_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_1_DepthwiseConv2d_bias": iris_Conv2D_block_1_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_1_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_1_DepthwiseConv2d_output),
        "iris_Conv2D_block_1_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_1_Conv2D_1_weight),
        "iris_Conv2D_block_1_Conv2D_1_bias": iris_Conv2D_block_1_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_1_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_1_Conv2D_1_output),
        "iris_Conv2D_block_1_output": tfLite_weight_to_torch(iris_Conv2D_block_1_output),

        "iris_Prelu_2_weight": iris_Prelu_2_weight.flatten(),
        "iris_Prelu_2_output": tfLite_weight_to_torch(iris_Prelu_2_output),
        "iris_Conv2D_block_v2_0_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_Conv2D_0_weight),
        "iris_Conv2D_block_v2_0_Conv2D_0_bias": iris_Conv2D_block_v2_0_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_v2_0_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_Conv2D_0_output),
        "iris_Conv2D_block_v2_0_Prelu_0_weight": iris_Conv2D_block_v2_0_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_v2_0_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_Prelu_0_output),
        "iris_Conv2D_block_v2_0_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_v2_0_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_v2_0_DepthwiseConv2d_bias": iris_Conv2D_block_v2_0_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_v2_0_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_DepthwiseConv2d_output),
        "iris_Conv2D_block_v2_0_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_Conv2D_1_weight),
        "iris_Conv2D_block_v2_0_Conv2D_1_bias": iris_Conv2D_block_v2_0_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_v2_0_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_Conv2D_1_output),
        "iris_Conv2D_block_v2_0_maxpool_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_maxpool_output),
        "iris_Conv2D_block_v2_0_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_0_output),

        "iris_Prelu_3_weight": iris_Prelu_3_weight.flatten(),
        "iris_Prelu_3_output": tfLite_weight_to_torch(iris_Prelu_3_output),
        "iris_Conv2D_block_2_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_2_Conv2D_0_weight),
        "iris_Conv2D_block_2_Conv2D_0_bias": iris_Conv2D_block_2_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_2_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_2_Conv2D_0_output),
        "iris_Conv2D_block_2_Prelu_0_weight": iris_Conv2D_block_2_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_2_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_2_Prelu_0_output),
        "iris_Conv2D_block_2_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_2_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_2_DepthwiseConv2d_bias": iris_Conv2D_block_2_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_2_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_2_DepthwiseConv2d_output),
        "iris_Conv2D_block_2_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_2_Conv2D_1_weight),
        "iris_Conv2D_block_2_Conv2D_1_bias": iris_Conv2D_block_2_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_2_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_2_Conv2D_1_output),
        "iris_Conv2D_block_2_output": tfLite_weight_to_torch(iris_Conv2D_block_2_output),

        "iris_Prelu_4_weight": iris_Prelu_4_weight.flatten(),
        "iris_Prelu_4_output": tfLite_weight_to_torch(iris_Prelu_4_output),
        "iris_Conv2D_block_3_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_3_Conv2D_0_weight),
        "iris_Conv2D_block_3_Conv2D_0_bias": iris_Conv2D_block_3_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_3_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_3_Conv2D_0_output),
        "iris_Conv2D_block_3_Prelu_0_weight": iris_Conv2D_block_3_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_3_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_3_Prelu_0_output),
        "iris_Conv2D_block_3_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_3_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_3_DepthwiseConv2d_bias": iris_Conv2D_block_3_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_3_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_3_DepthwiseConv2d_output),
        "iris_Conv2D_block_3_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_3_Conv2D_1_weight),
        "iris_Conv2D_block_3_Conv2D_1_bias": iris_Conv2D_block_3_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_3_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_3_Conv2D_1_output),
        "iris_Conv2D_block_3_output": tfLite_weight_to_torch(iris_Conv2D_block_3_output),

        "iris_Prelu_5_weight": iris_Prelu_5_weight.flatten(),
        "iris_Prelu_5_output": tfLite_weight_to_torch(iris_Prelu_5_output),
        "iris_Conv2D_block_v2_1_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_Conv2D_0_weight),
        "iris_Conv2D_block_v2_1_Conv2D_0_bias": iris_Conv2D_block_v2_1_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_v2_1_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_Conv2D_0_output),
        "iris_Conv2D_block_v2_1_Prelu_0_weight": iris_Conv2D_block_v2_1_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_v2_1_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_Prelu_0_output),
        "iris_Conv2D_block_v2_1_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_v2_1_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_v2_1_DepthwiseConv2d_bias": iris_Conv2D_block_v2_1_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_v2_1_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_DepthwiseConv2d_output),
        "iris_Conv2D_block_v2_1_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_Conv2D_1_weight),
        "iris_Conv2D_block_v2_1_Conv2D_1_bias": iris_Conv2D_block_v2_1_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_v2_1_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_Conv2D_1_output),
        "iris_Conv2D_block_v2_1_maxpool_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_maxpool_output),
        "iris_Conv2D_block_v2_1_output": tfLite_weight_to_torch(iris_Conv2D_block_v2_1_output),

        "iris_Prelu_6_weight": iris_Prelu_6_weight.flatten(),
        "iris_Prelu_6_output": tfLite_weight_to_torch(iris_Prelu_6_output),
        "iris_Conv2D_block_4_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_4_Conv2D_0_weight),
        "iris_Conv2D_block_4_Conv2D_0_bias": iris_Conv2D_block_4_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_4_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_4_Conv2D_0_output),
        "iris_Conv2D_block_4_Prelu_0_weight": iris_Conv2D_block_4_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_4_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_4_Prelu_0_output),
        "iris_Conv2D_block_4_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_4_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_4_DepthwiseConv2d_bias": iris_Conv2D_block_4_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_4_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_4_DepthwiseConv2d_output),
        "iris_Conv2D_block_4_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_4_Conv2D_1_weight),
        "iris_Conv2D_block_4_Conv2D_1_bias": iris_Conv2D_block_4_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_4_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_4_Conv2D_1_output),
        "iris_Conv2D_block_4_output": tfLite_weight_to_torch(iris_Conv2D_block_4_output),

        "iris_Prelu_7_weight": iris_Prelu_7_weight.flatten(),
        "iris_Prelu_7_output": tfLite_weight_to_torch(iris_Prelu_7_output),
        "iris_Conv2D_block_5_Conv2D_0_weight": tfLite_weight_to_torch(iris_Conv2D_block_5_Conv2D_0_weight),
        "iris_Conv2D_block_5_Conv2D_0_bias": iris_Conv2D_block_5_Conv2D_0_bias.flatten(),
        "iris_Conv2D_block_5_Conv2D_0_output": tfLite_weight_to_torch(iris_Conv2D_block_5_Conv2D_0_output),
        "iris_Conv2D_block_5_Prelu_0_weight": iris_Conv2D_block_5_Prelu_0_weight.flatten(),
        "iris_Conv2D_block_5_Prelu_0_output": tfLite_weight_to_torch(iris_Conv2D_block_5_Prelu_0_output),
        "iris_Conv2D_block_5_DepthwiseConv2d_weight": np.transpose(iris_Conv2D_block_5_DepthwiseConv2d_weight, (3,0,1,2)),
        "iris_Conv2D_block_5_DepthwiseConv2d_bias": iris_Conv2D_block_5_DepthwiseConv2d_bias.flatten(),
        "iris_Conv2D_block_5_DepthwiseConv2d_output": tfLite_weight_to_torch(iris_Conv2D_block_5_DepthwiseConv2d_output),
        "iris_Conv2D_block_5_Conv2D_1_weight": tfLite_weight_to_torch(iris_Conv2D_block_5_Conv2D_1_weight),
        "iris_Conv2D_block_5_Conv2D_1_bias": iris_Conv2D_block_5_Conv2D_1_bias.flatten(),
        "iris_Conv2D_block_5_Conv2D_1_output": tfLite_weight_to_torch(iris_Conv2D_block_5_Conv2D_1_output),
        "iris_Conv2D_block_5_output": tfLite_weight_to_torch(iris_Conv2D_block_5_output),

        "iris_Prelu_8_weight": iris_Prelu_8_weight.flatten(),
        "iris_Prelu_8_output": tfLite_weight_to_torch(iris_Prelu_8_output),
        "iris_Conv2D_out_weight": tfLite_weight_to_torch(iris_Conv2D_out_weight),
        "iris_Conv2D_out_bias": iris_Conv2D_out_bias.flatten(),
        "iris_Conv2D_out_output": tfLite_weight_to_torch(iris_Conv2D_out_output),
        "iris_output": output_data_iris.flatten() 
    }   

    with open("./data/weights.pkl", 'wb') as picklefile:
        pickle.dump(weights, picklefile)


    # print(interpreter.get_tensor(details[3]['index']).shape)
