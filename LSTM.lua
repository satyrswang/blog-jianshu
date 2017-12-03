require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size)
  --------------------- input structure ---------------------
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- network input
  table.insert(inputs, nn.Identity()())   -- c at time t-1
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  local input = inputs[1]
  local prev_c = inputs[2]
  local prev_h = inputs[3]

  --------------------- preactivations ----------------------
  local i2h = nn.Linear(input_size, 4 * rnn_size)(input)   -- input to hidden
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)    -- hidden to hidden
  local preactivations = nn.CAddTable()({i2h, h2h})        -- i2h + h2h

  ------------------ non-linear transforms ------------------
  -- gates
  local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
  local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

  -- input
  local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
  local in_transform = nn.Tanh()(in_chunk)

  ---------------------- gate narrows -----------------------
  local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
  local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
  local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)

  --------------------- next cell state ---------------------
  local c_forget = nn.CMulTable()({forget_gate, prev_c})  -- previous cell state contribution
  local c_input = nn.CMulTable()({in_gate, in_transform}) -- input contribution
  local next_c = nn.CAddTable()({
    c_forget,
    c_input
  })

  -------------------- next hidden state --------------------
  local c_transform = nn.Tanh()(next_c)
  local next_h = nn.CMulTable()({out_gate, c_transform})

  --------------------- output structure --------------------
  outputs = {}
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return LSTM



network = {LSTM.create(3, 4), LSTM.create(4, 4), LSTM.create(4, 3)}

-- network input
local x = torch.randn(1, 3)
local previous_state = {
  {torch.zeros(1, 4), torch.zeros(1,4)},
  {torch.zeros(1, 4), torch.zeros(1,4)},
  {torch.zeros(1, 3), torch.zeros(1,3)}
}

-- network output
output = nil
next_state = {}

-- forward pass
local layer_input = {x, table.unpack(previous_state[1])}
for l = 1, #network do
  -- forward the input
  local layer_output = network[l]:forward(layer_input)
  -- save output state for next iteration
  table.insert(next_state, layer_output)
  -- extract hidden state from output
  local layer_h = layer_output[2]
  -- prepare next layer's input or set the output
  if l < #network then
    layer_input = {layer_h, table.unpack(previous_state[l + 1])}
  else
    output = layer_h
  end
end

print(next_state)
print(output)
