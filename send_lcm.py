import argparse
import pydrake.systems.lcm as mut
from pydrake.lcm import DrakeLcm, DrakeMockLcm, Subscriber
from lcm import LCM
from drake import lcmt_letters

parser = argparse.ArgumentParser(description="send_lcm")
parser.add_argument("-x", default=0)
parser.add_argument("-y", default=0)
parser.add_argument("-l", default='A')
args = parser.parse_args()
def callback(self, channel, msg):
    print("entered")
    print(msg.x)
    print(msg.y)
    print(msg.letter)

def main():
    lcm = LCM()
    msg = lcmt_letters()
    print(args.x)
    msg.x = args.x
    msg.y = args[1]
    msg.letter = args[2]
    # lcm.subscribe("TEST_CHANNEL",callback)
    lcm.publish("TEST_CHANNEL",msg.encode())

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