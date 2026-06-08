#include "LoRaWan_APP.h"
#include "Arduino.h"
#include "HT_st7735.h"
#include "HT_TinyGPS++.h"
#include <Wire.h>

TinyGPSPlus GPS;
HT_st7735 st7735;

// ADXL345
#define ADXL345_ADDR 0x53
#define POWER_CTL    0x2D
#define DATA_FORMAT  0x31
#define DATAX0       0x32

uint16_t userChannelsMask[6] = { 0x00FF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 };

/* ABP parameters */
uint8_t nwkSKey[] = { 0x15, 0xb1, 0xd0, 0xef, 0xa4, 0x63, 0xdf, 0xbe, 0x3d, 0x11, 0x18, 0x1e, 0x1e, 0xc7, 0xda, 0x85 };
uint8_t appSKey[] = { 0xd7, 0x2c, 0x78, 0x75, 0x8c, 0xdc, 0xca, 0xbf, 0x55, 0xee, 0x4a, 0x77, 0x8d, 0x16, 0xef, 0x67 };
uint32_t devAddr = (uint32_t)0x007e6ae1;

uint8_t devEui[] = {0x11,0x21,0x22,0,0,0,0,0x12};
uint8_t appEui[] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
uint8_t appKey[] = {
    0x77,0x77,0x77,0x77,0x77,0x77,0x77,0x77,
    0x77,0x77,0x77,0x77,0x77,0x77,0x77,0x77
};

LoRaMacRegion_t loraWanRegion = LORAMAC_REGION_AS923;
DeviceClass_t   loraWanClass  = CLASS_A;
bool  overTheAirActivation = true;
bool  loraWanAdr           = true;
bool  isTxConfirmed        = false;
uint32_t appTxDutyCycle    = 15000;
uint8_t  appPort           = 2;
uint8_t  confirmedNbTrials = 4;

/* ── ADXL345 helpers ─────────────────────────────── */
void adxl345Init() {
    Wire.setClock(100000);
    delay(100);
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(POWER_CTL);
    Wire.write(0x08); // wake up, measurement mode
    Wire.endTransmission();
    delay(100);
}

bool adxl345Read(float &x, float &y, float &z) {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(DATAX0);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)ADXL345_ADDR, (uint8_t)6);

    if (Wire.available() < 6) return false;

    byte xl = Wire.read(), xh = Wire.read();
    byte yl = Wire.read(), yh = Wire.read();
    byte zl = Wire.read(), zh = Wire.read();

    x = (int16_t)((xh << 8) | xl) * 0.0039f;
    y = (int16_t)((yh << 8) | yl) * 0.0039f;
    z = (int16_t)((zh << 8) | zl) * 0.0039f;
    return true;
}

/* ── Build LoRa payload ──────────────────────────── */
static void prepareTxFrame(uint8_t port)
{
    pinMode(Vext, OUTPUT);
    digitalWrite(Vext, HIGH);  //ON OLED

    // --- GPS ---
    float lat = 0.0, lon = 0.0;
    uint32_t start = millis();
    Serial.println("Waiting GPS...");

    while (!GPS.location.isValid() && (millis() - start) < 10000) {
        st7735.st7735_fill_screen(ST7735_BLACK);
        st7735.st7735_write_str(0, 0, "GPS Reading..");
        while (Serial1.available()) GPS.encode(Serial1.read());
        st7735.st7735_fill_screen(ST7735_BLACK);
        st7735.st7735_write_str(0, 0, "NO GPS..");
        delay(1000);
    }


    if (GPS.location.isValid()) {
        lat = GPS.location.lat();
        lon = GPS.location.lng();
        Serial.print("LAT: "); Serial.println(lat, 6);
        Serial.print("LON: "); Serial.println(lon, 6);
    }

      // --- ADXL345 ---
    float ax = 0.0, ay = 0.0, az = 0.0;
    if (adxl345Read(ax, ay, az)) {
        Serial.print("X: "); Serial.print(ax, 2);
        Serial.print("  Y: "); Serial.print(ay, 2);
        Serial.print("  Z: "); Serial.println(az, 2);
    } else {
        Serial.println("ADXL345 read failed!");
    }

    // Display everything on one screen
    st7735.st7735_fill_screen(ST7735_BLACK);
    st7735.st7735_write_str(0, 0,  "Lat: " + String(lat, 4));
    st7735.st7735_write_str(0, 15, "Lon: " + String(lon, 4));
    st7735.st7735_write_str(0, 30, "X: " + String(ax, 2) + "g");
    st7735.st7735_write_str(0, 45, "Y: " + String(ay, 2) + "g");
    st7735.st7735_write_str(0, 60, "Z: " + String(az, 2) + "g");
    delay(1500);


    digitalWrite(Vext, LOW); //off OLED

    // --- Pack 14-byte payload ---
    // GPS: 2× int32  (lat/lon × 1e6)
    // Accel: 3× int16 (g × 100)
    int32_t lat_i = (int32_t)(lat * 1e6);
    int32_t lon_i = (int32_t)(lon * 1e6);
    int16_t ax_i  = (int16_t)(ax * 100);
    int16_t ay_i  = (int16_t)(ay * 100);
    int16_t az_i  = (int16_t)(az * 100);

    appDataSize = 0;
    // Latitude (4 bytes)
    appData[appDataSize++] = (lat_i >> 24) & 0xFF;
    appData[appDataSize++] = (lat_i >> 16) & 0xFF;
    appData[appDataSize++] = (lat_i >>  8) & 0xFF;
    appData[appDataSize++] =  lat_i        & 0xFF;
    // Longitude (4 bytes)
    appData[appDataSize++] = (lon_i >> 24) & 0xFF;
    appData[appDataSize++] = (lon_i >> 16) & 0xFF;
    appData[appDataSize++] = (lon_i >>  8) & 0xFF;
    appData[appDataSize++] =  lon_i        & 0xFF;
    // X accel (2 bytes)
    appData[appDataSize++] = (ax_i >> 8) & 0xFF;
    appData[appDataSize++] =  ax_i       & 0xFF;
    // Y accel (2 bytes)
    appData[appDataSize++] = (ay_i >> 8) & 0xFF;
    appData[appDataSize++] =  ay_i       & 0xFF;
    // Z accel (2 bytes)
    appData[appDataSize++] = (az_i >> 8) & 0xFF;
    appData[appDataSize++] =  az_i       & 0xFF;

}

/* ── Setup & Loop (unchanged structure) ─────────── */
void setup()
{
    Serial.begin(115200);
    Serial1.begin(115200, SERIAL_8N1, 33, 34);

    Wire.begin(6,7);

    pinMode(Vext, OUTPUT);
    digitalWrite(Vext, HIGH);

    st7735.st7735_init();
    st7735.st7735_fill_screen(ST7735_BLACK);
    st7735.st7735_write_str(0, 0, "Starting...");

    adxl345Init();

    Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);
}

void loop()
{
    switch (deviceState)
    {
        case DEVICE_STATE_INIT:
            LoRaWAN.init(loraWanClass, loraWanRegion);
            LoRaWAN.setDefaultDR(3);
            deviceState = DEVICE_STATE_JOIN;
            break;

        case DEVICE_STATE_JOIN:
            st7735.st7735_fill_screen(ST7735_BLACK);
            st7735.st7735_write_str(0, 0, "Joining TTN...");
            Serial.println("Joining TTN...");
            LoRaWAN.join();
            break;

        case DEVICE_STATE_SEND:
            Serial.println("JOIN SUCCESSFUL");
            st7735.st7735_fill_screen(ST7735_BLACK);
            st7735.st7735_write_str(0, 0, "JOINED");
            prepareTxFrame(appPort);
            Serial.println("Sending...");
            LoRaWAN.send();
            deviceState = DEVICE_STATE_CYCLE;
            break;

        case DEVICE_STATE_CYCLE:
            txDutyCycleTime = appTxDutyCycle;
            LoRaWAN.cycle(txDutyCycleTime);
            deviceState = DEVICE_STATE_SLEEP;
            break;

        case DEVICE_STATE_SLEEP:
            LoRaWAN.sleep(loraWanClass);
            break;

        default:
            deviceState = DEVICE_STATE_INIT;
            break;
    }
}