using Agora_RTC_Plugin.API_Example.Examples.Basic.JoinChannelAudio;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class dog2 : MonoBehaviour
{
    // Start is called before the first frame update

    private SpriteRenderer spriteRenderer;

    // ��Start�����л�ȡSpriteRenderer���
    void Start()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();

        // ȷ��SpriteRenderer�������
        if (spriteRenderer != null)
        {
            // ��SpriteRenderer��Ϊ���ɼ�
            spriteRenderer.enabled = false;
        }
        else
        {
            Debug.LogError("û���ҵ�SpriteRenderer���");
        }
    }

    private void Update()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();


        if (JoinChannelAudio.state == 2)
        {

            spriteRenderer.enabled = true;
        }
        else
            spriteRenderer.enabled = false;
    }
}
