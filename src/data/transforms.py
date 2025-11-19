from torchvision import transforms

def get_train_transforms(input_size):
    '''
    Very basic train transforms for the baseline
    Is center frame, so we rely mostly on spatial augmentation
    '''
    
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random',
            inplace=True,
        ),
    ])
    
def get_val_transforms(input_size):
    '''
    Very basic val transforms for the baseline
    Is center frame, so we rely mostly on spatial augmentation
    '''
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])