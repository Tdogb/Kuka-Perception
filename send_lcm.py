import argparse
import pydrake.systems.lcm as mut
# from pydrake.lcm import DrakeLcm, DrakeMockLcm, Subscriber
from lcm import LCM
from drake import lcmt_letters
parser = argparse.ArgumentParser(description="send_lcm")
parser.add_argument("-x", default=2, type=float)
parser.add_argument("-y", default=1, type=float)
parser.add_argument("-l", default='B', type=str)
parser.add_argument("-i", default=0, type=int)

args = parser.parse_args()
lcm_ = LCM()
def callback(channel, msg):
    print("entered")
    newMsgC = lcmt_letters()
    newMsg = newMsgC.decode(msg)
    print(newMsg.x)
    print(newMsg.y)
    print(newMsg.letter)

def main():
    lcm_.subscribe("TEST_CHANNEL",callback)
    # print("Run")
    msg = lcmt_letters()
    msg.x = args.x
    msg.y = args.y
    msg.letter = args.l
    msg.index = args.i
    lcm_.publish("TEST_CHANNEL",msg.encode())
    lcm_.handle()

if __name__ == "__main__":
    main()

# lcm = DrakeMockLcm()
# dut = mut.LcmPublisherSystem.Make(
#             channel="TEST_CHANNEL", lcm_type=str, lcm=lcm,
#             publish_period=0.1)
# subscriber = Subscriber(lcm, "TEST_CHANNEL", str)
# # model_message = self._model_message()
# # self._fix_and_publish(dut, AbstractValue.Make(model_message))
# dut.Publish("Hello World")
# lcm.HandleSubscriptions(0)
# print(args)

# def _fix_and_publish(self, dut, value):
#     context = dut.CreateDefaultContext()
#     dut.get_input_port(0).FixValue(context, value)
#     dut.Publish(context)