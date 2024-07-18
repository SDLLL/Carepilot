using Agora_RTC_Plugin.API_Example.Examples.Basic.JoinChannelAudio;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class dog : MonoBehaviour
{
    private SpriteRenderer spriteRenderer;

    // 在Start方法中获取SpriteRenderer组件
    void Start()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();

        // 确保SpriteRenderer组件存在
        if (spriteRenderer != null)
        {
            // 将SpriteRenderer设为不可见
            spriteRenderer.enabled = false;
        }
        else
        {
            Debug.LogError("没有找到SpriteRenderer组件");
        }
    }

    private void Update()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();


        if (JoinChannelAudio.state == 1)
        {
            
            spriteRenderer.enabled = true;
        }
        else
            spriteRenderer.enabled = false;
    }
}
