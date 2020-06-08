#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
from CorpusGenerator import CorpusGenerator

class ASRController():

    def __init__(self, corpGen):
        rospy.init_node('asr_controller')
        self.corpGen = corpGen
        self.soundhandle = SoundClient()
        self.voice = 'voice_kal_diphone'
        self.volume = 1.0


        rospy.Subscriber('/grammar_data', String, self.parse_speech)
        rospy.spin()

    def parse_speech(self, speech_data):
        if speech_data.data in self.corpGen.listQuestions():
            answer = self.corpGen.getAnswer(speech_data.data)
            self.say(answer)

    def say(self, speech):
        self.soundhandle.say(speech, self.voice, self.volume)

if __name__ == "__main__":
    namesFile = '../asr/resources/Names.xml'
    objectsFile = '../asr/resources/Objects.xml'
    locationsFile = '../asr/resources/Locations.xml' 
    gesturesFile = '../asr/resources/Gestures.xml'
    questionsFile = '../asr/resources/Questions.xml'
    
    corpGen = CorpusGenerator()
    ASRController(corpGen)
