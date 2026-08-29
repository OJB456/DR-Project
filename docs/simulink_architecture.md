# Simulink architecture specification

MATLAB/Simulink was not assumed or emulated by this Python prototype. If it is
available for the SIH system-level demonstration, construct a discrete-event
or rate-based model with these blocks:

```text
Rural PHC -> Fundus Camera -> Image Quality Gate -> Local AI Inference
          -> Referral Decision -> Specialist Queue -> Ophthalmologist
```

Suggested tunable workspace variables:

- `patient_arrival_rate_per_hour`
- `image_acquisition_time_seconds`
- `quality_rejection_rate`
- `ai_inference_time_seconds`
- `network_delay_seconds`
- `specialist_service_rate_per_hour`
- `referral_rate`
- `offline_sync_delay_seconds`

The quality gate should route rejected captures back to **Fundus Camera**.
Accepted captures should route to local inference; `referable == true` or low
confidence should enter the specialist queue. Non-referable results can be
held locally and synchronized later when connectivity returns. The Python app
is fully functional without this optional MATLAB model.
