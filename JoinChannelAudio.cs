using System;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Serialization;
using Agora.Rtc;
using io.agora.rtc.demo;
using System.Collections;
using System.Text;
using System.Text.RegularExpressions;
using TMPro;
using UnityEngine.Android;
using UnityEngine.SceneManagement;
using Rokid.UXR.Module;
using Unity.VisualScripting;

namespace Agora_RTC_Plugin.API_Example.Examples.Basic.JoinChannelAudio
{

    public class JoinChannelAudio : MonoBehaviour
    {
        public static int state = 0;
        

        private bool isInit = false;
        

        [FormerlySerializedAs("appIdInput")]
        [SerializeField]
        private AppIdInput _appIdInput;

        [Header("_____________Basic Configuration_____________")]
        [FormerlySerializedAs("APP_ID")]
        [SerializeField]
        private string _appID = "";

        [FormerlySerializedAs("TOKEN")]
        [SerializeField]
        private string _token = "";

        [FormerlySerializedAs("CHANNEL_NAME")]
        [SerializeField]
        private string _channelName = "";

        public Text LogText;
        
        public Text InputText;

        public Text OutputText;

        internal Logger Log;
        internal IRtcEngine RtcEngine = null;

        private IAudioDeviceManager _audioDeviceManager;
        private DeviceInfo[] _audioPlaybackDeviceInfos;
        public Dropdown _audioDeviceSelect;

        private void Awake()
        {
            //Permission.RequestUserPermission("android.permission.RECORD_AUDIO");
            //state = 0;

            if (!Permission.HasUserAuthorizedPermission("android.permission.RECORD_AUDIO"))
            {
                Permission.RequestUserPermission("android.permission.RECORD_AUDIO");
            }

        }

        // Start is called before the first frame update
        private void Start()
        {

            if (!Permission.HasUserAuthorizedPermission("android.permission.RECORD_AUDIO"))
            {
                Debug.LogError("-RKX- UXR-Sample:: no permission RECORD_AUDIO, will return!!! ");
                SceneManager.LoadScene(0);//WA
                return;
            }

            initVoice();
            //OfflineVoiceModule.Instance.AddInstruct(LANGUAGE.ENGLISH, "Show blue", "show blue", this.gameObject.name, "OnReceive");
            OfflineVoiceModule.Instance.AddInstruct(LANGUAGE.CHINESE, "小顺", "xiao shun", this.gameObject.name, "OnReceive");
            OfflineVoiceModule.Instance.AddInstruct(LANGUAGE.CHINESE, "停止", "tin zhi", this.gameObject.name, "OnReceive");

            OfflineVoiceModule.Instance.Commit();

            LoadAssetData();
            if (CheckAppId())
            {
                InitRtcEngine();
                SetBasicConfiguration();
            }

#if UNITY_IOS || UNITY_ANDROID
            var text = GameObject.Find("Canvas/Scroll View/Viewport/Content/AudioDeviceManager").GetComponent<Text>();
            text.text = "Audio device manager not support in this platform";

            GameObject.Find("Canvas/Scroll View/Viewport/Content/AudioDeviceButton").SetActive(false);
            GameObject.Find("Canvas/Scroll View/Viewport/Content/deviceIdSelect").SetActive(false);
            GameObject.Find("Canvas/Scroll View/Viewport/Content/AudioSelectButton").SetActive(false);
#endif

        }


        private void Update()
        {
            PermissionHelper.RequestMicrophontPermission();
            LogText.text = string.Format("{0}", state); 
        }

        private bool CheckAppId()
        {
            Log = new Logger(LogText);
            return Log.DebugAssert(_appID.Length > 10, "Please fill in your appId in API-Example/profile/appIdInput.asset!!!!!");
        }

        //Show data in AgoraBasicProfile
        [ContextMenu("ShowAgoraBasicProfileData")]
        private void LoadAssetData()
        {
            if (_appIdInput == null) return;
            _appID = _appIdInput.appID;
            _token = _appIdInput.token;
            _channelName = _appIdInput.channelName;
        }

        private void InitRtcEngine()
        {
            RtcEngine = Agora.Rtc.RtcEngine.CreateAgoraRtcEngine();
            UserEventHandler handler = new UserEventHandler(this,InputText,OutputText);
            RtcEngineContext context = new RtcEngineContext();
            context.appId = _appID;
            context.channelProfile = CHANNEL_PROFILE_TYPE.CHANNEL_PROFILE_LIVE_BROADCASTING;
            context.audioScenario = AUDIO_SCENARIO_TYPE.AUDIO_SCENARIO_DEFAULT;
            context.areaCode = AREA_CODE.AREA_CODE_GLOB;

            RtcEngine.Initialize(context);
            RtcEngine.InitEventHandler(handler);
        }

        private void SetBasicConfiguration()
        {
            RtcEngine.EnableAudio();
            RtcEngine.SetChannelProfile(CHANNEL_PROFILE_TYPE.CHANNEL_PROFILE_COMMUNICATION);
            RtcEngine.SetClientRole(CLIENT_ROLE_TYPE.CLIENT_ROLE_BROADCASTER);
        }

        void OnReceive(string msg)
        {
            Debug.Log("-RKX- UXR-Sample:: On Voice Response received : " + msg);
            //InfoText.text = msg;

            if (string.Equals("小顺", msg))
            {
                InputText.text = "小顺已被唤起";
                JoinChannel();
                state = 1;

            }
            else if(string.Equals("停止", msg))
            {
                InputText.text = "小顺已休眠";
                LeaveChannel();
                state = 0;

            }
        }

        private void initVoice()
        {
            // Start plugin `VoiceControlFragment` , init once.
            if (!isInit)
            {
                Debug.Log("-RKX- UXR-Sample start init voice.");
                ModuleManager.Instance.RegistModule("com.rokid.voicecommand.VoiceCommandHelper", false);

                //Should choose one of the language to use
                OfflineVoiceModule.Instance.ChangeVoiceCommandLanguage(LANGUAGE.CHINESE); //Support for CHINESE.
                //OfflineVoiceModule.Instance.ChangeVoiceCommandLanguage(LANGUAGE.ENGLISH); //Support for ENGLISH.

                isInit = true;
            }
        }

        #region -- Button Events ---

        public void StartEchoTest()
        {
            EchoTestConfiguration config = new EchoTestConfiguration();
            config.intervalInSeconds = 2;
            config.enableAudio = true;
            config.enableVideo = false;
            config.token = this._appID;
            config.channelId = "echo_test_channel";
            RtcEngine.StartEchoTest(config);
            Log.UpdateLog("StartEchoTest, speak now. You cannot conduct another echo test or join a channel before StopEchoTest");
        }

        public void StopEchoTest()
        {
            RtcEngine.StopEchoTest();
        }

        public void JoinChannel()
        {
            RtcEngine.JoinChannel(_token, _channelName, "", 666);
        }

        public void LeaveChannel()
        {
            RtcEngine.LeaveChannel();
        }

        public void StopPublishAudio()
        {
            var options = new ChannelMediaOptions();
            options.publishMicrophoneTrack.SetValue(false);
            var nRet = RtcEngine.UpdateChannelMediaOptions(options);
            this.Log.UpdateLog("UpdateChannelMediaOptions: " + nRet);
        }

        public void StartPublishAudio()
        {
            var options = new ChannelMediaOptions();
            options.publishMicrophoneTrack.SetValue(true);
            var nRet = RtcEngine.UpdateChannelMediaOptions(options);
            this.Log.UpdateLog("UpdateChannelMediaOptions: " + nRet);
        }

        public void GetAudioPlaybackDevice()
        {
            _audioDeviceSelect.ClearOptions();
            _audioDeviceManager = RtcEngine.GetAudioDeviceManager();
            _audioPlaybackDeviceInfos = _audioDeviceManager.EnumeratePlaybackDevices();
            Log.UpdateLog(string.Format("AudioPlaybackDevice count: {0}", _audioPlaybackDeviceInfos.Length));
            for (var i = 0; i < _audioPlaybackDeviceInfos.Length; i++)
            {
                Log.UpdateLog(string.Format("AudioPlaybackDevice device index: {0}, name: {1}, id: {2}", i,
                    _audioPlaybackDeviceInfos[i].deviceName, _audioPlaybackDeviceInfos[i].deviceId));
            }

            _audioDeviceSelect.AddOptions(_audioPlaybackDeviceInfos.Select(w =>
                    new Dropdown.OptionData(
                        string.Format("{0} :{1}", w.deviceName, w.deviceId)))
                .ToList());
        }

        public void SelectAudioPlaybackDevice()
        {
            if (_audioDeviceSelect == null) return;
            var option = _audioDeviceSelect.options[_audioDeviceSelect.value].text;
            if (string.IsNullOrEmpty(option)) return;

            var deviceId = option.Split(":".ToCharArray(), StringSplitOptions.RemoveEmptyEntries)[1];
            var ret = _audioDeviceManager.SetPlaybackDevice(deviceId);
            Log.UpdateLog("SelectAudioPlaybackDevice ret:" + ret + " , DeviceId: " + deviceId);
        }

        #endregion

        private void OnDestroy()
        {
            Debug.Log("OnDestroy");
            if (RtcEngine == null) return;
            RtcEngine.InitEventHandler(null);
            RtcEngine.LeaveChannel();
            RtcEngine.Dispose();
        }
    }

    public class CallBackData
    {
        public string uid;
        public string text;
        public string is_final;
        public string type;
    }

    #region -- Agora Event ---

    internal class UserEventHandler : IRtcEngineEventHandler
    {
        private readonly JoinChannelAudio _audioSample;
        private Text _inputtext;
        private Text _outputtext;

        internal UserEventHandler(JoinChannelAudio audioSample,Text InputText,Text OutputText)
        {

            _audioSample = audioSample;
            _inputtext = InputText;
            _outputtext = OutputText;

        }

        public override void OnError(int err, string msg)
        {
            _audioSample.Log.UpdateLog(string.Format("OnError err: {0}, msg: {1}", err, msg));
        }

        public override void OnJoinChannelSuccess(RtcConnection connection, int elapsed)
        {
            int build = 0;
            _audioSample.Log.UpdateLog(string.Format("sdk version: ${0}",
                _audioSample.RtcEngine.GetVersion(ref build)));
            _audioSample.Log.UpdateLog(
                string.Format("OnJoinChannelSuccess channelName: {0}, uid: {1}, elapsed: {2}",
                                connection.channelId, connection.localUid, elapsed));
        }

        public override void OnRejoinChannelSuccess(RtcConnection connection, int elapsed)
        {
            _audioSample.Log.UpdateLog("OnRejoinChannelSuccess");
        }

        public override void OnStreamMessage(RtcConnection connection, uint remoteUid, int streamId, byte[] data, ulong length, ulong sentTs)
        {
            base.OnStreamMessage(connection, remoteUid, streamId, data, length, sentTs);
            string utf8String = Encoding.UTF8.GetString(data);
            CallBackData json = JsonUtility.FromJson<CallBackData>(utf8String);
            if (json.type.Equals("input"))
            {
                _inputtext.text = Regex.Unescape(json.text);
                
                if (json.is_final.Equals("true")) JoinChannelAudio.state = 2;
            }
            else
            {
                _outputtext.text = Regex.Unescape(json.text);
                
                if (json.is_final.Equals("true")) JoinChannelAudio.state = 3;
            }

        }
        

        public override void OnLeaveChannel(RtcConnection connection, RtcStats stats)
        {
            _audioSample.Log.UpdateLog("OnLeaveChannel");
        }

        public override void OnClientRoleChanged(RtcConnection connection, CLIENT_ROLE_TYPE oldRole, CLIENT_ROLE_TYPE newRole, ClientRoleOptions newRoleOptions)
        {
            _audioSample.Log.UpdateLog("OnClientRoleChanged");
        }

        public override void OnUserJoined(RtcConnection connection, uint uid, int elapsed)
        {
            _audioSample.Log.UpdateLog(string.Format("OnUserJoined uid: ${0} elapsed: ${1}", uid, elapsed));
        }

        public override void OnUserOffline(RtcConnection connection, uint uid, USER_OFFLINE_REASON_TYPE reason)
        {
            _audioSample.Log.UpdateLog(string.Format("OnUserOffLine uid: ${0}, reason: ${1}", uid,
                (int)reason));
        }
    }

    #endregion
}